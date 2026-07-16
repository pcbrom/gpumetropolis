//! Native code generation for the CPU log-likelihood.
//!
//! The bytecode of a model's per-observation log-likelihood is translated to
//! Cranelift IR and JIT-compiled to a native function once per model. The CPU
//! backend then evaluates the log-likelihood as compiled code with the loop
//! over observations inside it, instead of interpreting the bytecode per
//! observation. This closes the gap against competitors that compile the
//! model.
//!
//! The CPU path works in f64: the log-density sums many terms, and f32 loses
//! the small difference that drives the Metropolis acceptance once the data
//! set is large. The GPU kernels stay in f32 with a tree reduction.
//!
//! The generated function has the C ABI
//! `fn(params: *const f64, data: *const f64, n_obs: i64, n_cols: i64) -> f64`
//! and returns the sum over observations of the per-observation expression.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

extern "C" fn jit_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn jit_log(x: f64) -> f64 {
    x.ln()
}
extern "C" fn jit_pow(b: f64, e: f64) -> f64 {
    b.powf(e)
}

/// A JIT-compiled log-likelihood. The native function is valid while the
/// `JITModule` it lives in is kept alive.
pub struct JitLoglik {
    _module: JITModule,
    func: extern "C" fn(*const f64, *const f64, i64, i64) -> f64,
}

impl JitLoglik {
    /// Sum of the per-observation log-likelihood over `n_obs` observations for
    /// one parameter vector.
    #[inline]
    pub fn eval(&self, params: &[f64], data: &[f64], n_obs: usize,
                n_cols: usize) -> f64 {
        (self.func)(params.as_ptr(), data.as_ptr(), n_obs as i64,
                    n_cols as i64)
    }
}

// SAFETY: the function is a bare native function over raw pointers with no
// shared mutable state; calling it from several threads is sound.
unsafe impl Send for JitLoglik {}
unsafe impl Sync for JitLoglik {}

/// A JIT-compiled log-likelihood gradient: one native function returns the
/// value summed over observations and accumulates the parameter gradient
/// into a caller-supplied buffer. The reverse-mode tape is unrolled at
/// compile time: the bytecode is static per model, so the forward values and
/// the backward adjoints become straight-line SSA code with no interpreter
/// dispatch and no runtime tape.
pub struct JitGrad {
    _module: JITModule,
    func: extern "C" fn(*const f64, *const f64, i64, i64, *mut f64) -> f64,
    n_params: usize,
}

impl JitGrad {
    /// Value of the summed log-likelihood; `grad` is overwritten with its
    /// gradient in the parameters.
    #[inline]
    pub fn eval_grad(&self, params: &[f64], data: &[f64], n_obs: usize,
                     n_cols: usize, grad: &mut [f64]) -> f64 {
        for slot in grad[..self.n_params].iter_mut() {
            *slot = 0.0;
        }
        (self.func)(params.as_ptr(), data.as_ptr(), n_obs as i64,
                    n_cols as i64, grad.as_mut_ptr())
    }
}

// SAFETY: as for JitLoglik, a bare native function over raw pointers.
unsafe impl Send for JitGrad {}
unsafe impl Sync for JitGrad {}

/// Compile the reverse-mode gradient of a per-observation log-likelihood to
/// native code. The generated function has the C ABI
/// `fn(params, data, n_obs, n_cols, grad) -> f64`: it loops the observations,
/// runs the unrolled forward pass, then the unrolled backward pass, and
/// accumulates `d(sum)/d(params[k])` into `grad[k]` (which the wrapper
/// zeroes). Falls back to an error string when the module cannot be built.
pub fn compile_grad(code: &[u32], consts: &[f64], n_params: usize)
                    -> Result<JitGrad, String> {
    if code.is_empty() {
        return Err("empty program".to_string());
    }
    let mut jb = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| e.to_string())?;
    jb.symbol("gpum_exp", jit_exp as *const u8);
    jb.symbol("gpum_log", jit_log as *const u8);
    jb.symbol("gpum_pow", jit_pow as *const u8);
    let mut module = JITModule::new(jb);

    let mut sig_un = module.make_signature();
    sig_un.params.push(AbiParam::new(types::F64));
    sig_un.returns.push(AbiParam::new(types::F64));
    let mut sig_bin = module.make_signature();
    sig_bin.params.push(AbiParam::new(types::F64));
    sig_bin.params.push(AbiParam::new(types::F64));
    sig_bin.returns.push(AbiParam::new(types::F64));

    let exp_id = module
        .declare_function("gpum_exp", Linkage::Import, &sig_un)
        .map_err(|e| e.to_string())?;
    let log_id = module
        .declare_function("gpum_log", Linkage::Import, &sig_un)
        .map_err(|e| e.to_string())?;
    let pow_id = module
        .declare_function("gpum_pow", Linkage::Import, &sig_bin)
        .map_err(|e| e.to_string())?;

    let mut sig = module.make_signature();
    for _ in 0..5 {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::F64));
    let func_id = module
        .declare_function("loglik_grad", Linkage::Local, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fb_ctx = FunctionBuilderContext::new();

    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let exp = module.declare_func_in_func(exp_id, b.func);
        let log = module.declare_func_in_func(log_id, b.func);
        let pow = module.declare_func_in_func(pow_id, b.func);

        let entry = b.create_block();
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.append_block_param(header, types::F64);
        b.append_block_param(header, types::I64);
        b.append_block_param(body, types::F64);
        b.append_block_param(body, types::I64);
        b.append_block_param(exit, types::F64);

        b.switch_to_block(entry);
        b.seal_block(entry);
        let p_params = b.block_params(entry)[0];
        let p_data = b.block_params(entry)[1];
        let n_obs = b.block_params(entry)[2];
        let n_cols = b.block_params(entry)[3];
        let p_grad = b.block_params(entry)[4];
        let zero_acc = b.ins().f64const(0.0f64);
        let zero_i = b.ins().iconst(types::I64, 0);
        b.ins().jump(header, &[zero_acc.into(), zero_i.into()]);

        b.switch_to_block(header);
        let acc = b.block_params(header)[0];
        let idx = b.block_params(header)[1];
        let cond = b.ins().icmp(IntCC::SignedLessThan, idx, n_obs);
        b.ins().brif(cond, body, &[acc.into(), idx.into()], exit,
                     &[acc.into()]);

        b.switch_to_block(body);
        let acc_b = b.block_params(body)[0];
        let idx_b = b.block_params(body)[1];
        let row_elems = b.ins().imul(idx_b, n_cols);
        let row_bytes = b.ins().imul_imm(row_elems, 8);
        let row_addr = b.ins().iadd(p_data, row_bytes);

        // Forward pass, unrolled: `vals[pc]` is the SSA value the
        // instruction produced; the operand tape indices are known at
        // compile time from a simulated index stack.
        let n_instr = code.len() / 2;
        let mut vals: Vec<Value> = Vec::with_capacity(n_instr);
        let mut lhs: Vec<usize> = vec![usize::MAX; n_instr];
        let mut rhs: Vec<usize> = vec![usize::MAX; n_instr];
        let mut istack: Vec<usize> = Vec::new();
        for pc in 0..n_instr {
            let op = code[2 * pc];
            let arg = code[2 * pc + 1] as usize;
            let v = match op {
                0 => {
                    istack.push(pc);
                    b.ins().f64const(consts[arg])
                }
                1 => {
                    istack.push(pc);
                    b.ins().load(types::F64, MemFlags::trusted(), p_params,
                                 (arg * 8) as i32)
                }
                2 => {
                    istack.push(pc);
                    b.ins().load(types::F64, MemFlags::trusted(), row_addr,
                                 (arg * 8) as i32)
                }
                7..=10 => {
                    let l = istack.pop().unwrap();
                    lhs[pc] = l;
                    istack.push(pc);
                    let x = vals[l];
                    match op {
                        7 => b.ins().fneg(x),
                        8 => {
                            let call = b.ins().call(exp, &[x]);
                            b.inst_results(call)[0]
                        }
                        9 => {
                            let call = b.ins().call(log, &[x]);
                            b.inst_results(call)[0]
                        }
                        _ => b.ins().sqrt(x),
                    }
                }
                _ => {
                    let r = istack.pop().unwrap();
                    let l = istack.pop().unwrap();
                    lhs[pc] = l;
                    rhs[pc] = r;
                    istack.push(pc);
                    let a = vals[l];
                    let bb = vals[r];
                    match op {
                        3 => b.ins().fadd(a, bb),
                        4 => b.ins().fsub(a, bb),
                        5 => b.ins().fmul(a, bb),
                        6 => b.ins().fdiv(a, bb),
                        _ => {
                            let call = b.ins().call(pow, &[a, bb]);
                            b.inst_results(call)[0]
                        }
                    }
                }
            };
            vals.push(v);
        }

        // Backward pass, unrolled: adjoints as SSA values, merged with fadd
        // as contributions arrive; parameter adjoints accumulate into the
        // grad buffer in memory (load, add, store), which carries them
        // across the observation loop.
        let mut adj: Vec<Option<Value>> = vec![None; n_instr];
        let one = b.ins().f64const(1.0f64);
        adj[n_instr - 1] = Some(one);
        for pc in (0..n_instr).rev() {
            let g = match adj[pc] {
                Some(v) => v,
                None => continue,
            };
            let op = code[2 * pc];
            let arg = code[2 * pc + 1] as usize;
            let mut bump = |b: &mut FunctionBuilder, slot: &mut Option<Value>,
                            v: Value| {
                *slot = Some(match *slot {
                    Some(old) => b.ins().fadd(old, v),
                    None => v,
                });
            };
            match op {
                0 | 2 => {}
                1 => {
                    let cur = b.ins().load(types::F64, MemFlags::trusted(),
                                           p_grad, (arg * 8) as i32);
                    let newv = b.ins().fadd(cur, g);
                    b.ins().store(MemFlags::trusted(), newv, p_grad,
                                  (arg * 8) as i32);
                }
                3 => {
                    let (l, r) = (lhs[pc], rhs[pc]);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, g);
                    adj[l] = a;
                    let mut a2 = adj[r].take();
                    bump(&mut b, &mut a2, g);
                    adj[r] = a2;
                }
                4 => {
                    let (l, r) = (lhs[pc], rhs[pc]);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, g);
                    adj[l] = a;
                    let neg = b.ins().fneg(g);
                    let mut a2 = adj[r].take();
                    bump(&mut b, &mut a2, neg);
                    adj[r] = a2;
                }
                5 => {
                    let (l, r) = (lhs[pc], rhs[pc]);
                    let gl = b.ins().fmul(g, vals[r]);
                    let gr = b.ins().fmul(g, vals[l]);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, gl);
                    adj[l] = a;
                    let mut a2 = adj[r].take();
                    bump(&mut b, &mut a2, gr);
                    adj[r] = a2;
                }
                6 => {
                    let (l, r) = (lhs[pc], rhs[pc]);
                    let gl = b.ins().fdiv(g, vals[r]);
                    let t = b.ins().fdiv(vals[pc], vals[r]);
                    let gr0 = b.ins().fmul(g, t);
                    let gr = b.ins().fneg(gr0);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, gl);
                    adj[l] = a;
                    let mut a2 = adj[r].take();
                    bump(&mut b, &mut a2, gr);
                    adj[r] = a2;
                }
                7 => {
                    let l = lhs[pc];
                    let neg = b.ins().fneg(g);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, neg);
                    adj[l] = a;
                }
                8 => {
                    let l = lhs[pc];
                    let gl = b.ins().fmul(g, vals[pc]);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, gl);
                    adj[l] = a;
                }
                9 => {
                    let l = lhs[pc];
                    let gl = b.ins().fdiv(g, vals[l]);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, gl);
                    adj[l] = a;
                }
                10 => {
                    let l = lhs[pc];
                    let half = b.ins().f64const(0.5f64);
                    let gh = b.ins().fmul(g, half);
                    let gl = b.ins().fdiv(gh, vals[pc]);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, gl);
                    adj[l] = a;
                }
                _ => {
                    let (l, r) = (lhs[pc], rhs[pc]);
                    // d/da a^b = b * a^(b-1)
                    let bm1 = {
                        let onec = b.ins().f64const(1.0f64);
                        b.ins().fsub(vals[r], onec)
                    };
                    let call = b.ins().call(pow, &[vals[l], bm1]);
                    let apow = b.inst_results(call)[0];
                    let gb = b.ins().fmul(vals[r], apow);
                    let gl = b.ins().fmul(g, gb);
                    let mut a = adj[l].take();
                    bump(&mut b, &mut a, gl);
                    adj[l] = a;
                    // d/db a^b = a^b ln(a), guarded to zero when a <= 0.
                    let calln = b.ins().call(log, &[vals[l]]);
                    let lna = b.inst_results(calln)[0];
                    let gr0 = b.ins().fmul(vals[pc], lna);
                    let gr1 = b.ins().fmul(g, gr0);
                    let zeroc = b.ins().f64const(0.0f64);
                    let pos = b.ins().fcmp(FloatCC::GreaterThan, vals[l],
                                           zeroc);
                    let gr = b.ins().select(pos, gr1, zeroc);
                    let mut a2 = adj[r].take();
                    bump(&mut b, &mut a2, gr);
                    adj[r] = a2;
                }
            }
        }

        let contrib = vals[n_instr - 1];
        let acc_new = b.ins().fadd(acc_b, contrib);
        let idx_new = b.ins().iadd_imm(idx_b, 1);
        b.ins().jump(header, &[acc_new.into(), idx_new.into()]);

        b.seal_block(header);
        b.seal_block(body);

        b.switch_to_block(exit);
        b.seal_block(exit);
        let result = b.block_params(exit)[0];
        b.ins().return_(&[result]);

        b.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| e.to_string())?;

    let ptr = module.get_finalized_function(func_id);
    // SAFETY: `ptr` is the compiled `loglik_grad` with exactly this C ABI.
    let func: extern "C" fn(*const f64, *const f64, i64, i64, *mut f64)
        -> f64 = unsafe { core::mem::transmute(ptr) };

    Ok(JitGrad { _module: module, func, n_params })
}

/// Compile a per-observation log-likelihood bytecode to native code. Returns an
/// error string when the JIT cannot be built; the caller falls back to the
/// interpreter.
pub fn compile_loglik(code: &[u32], consts: &[f64]) -> Result<JitLoglik, String> {
    let mut jb = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| e.to_string())?;
    jb.symbol("gpum_exp", jit_exp as *const u8);
    jb.symbol("gpum_log", jit_log as *const u8);
    jb.symbol("gpum_pow", jit_pow as *const u8);
    let mut module = JITModule::new(jb);

    let mut sig_un = module.make_signature();
    sig_un.params.push(AbiParam::new(types::F64));
    sig_un.returns.push(AbiParam::new(types::F64));
    let mut sig_bin = module.make_signature();
    sig_bin.params.push(AbiParam::new(types::F64));
    sig_bin.params.push(AbiParam::new(types::F64));
    sig_bin.returns.push(AbiParam::new(types::F64));

    let exp_id = module
        .declare_function("gpum_exp", Linkage::Import, &sig_un)
        .map_err(|e| e.to_string())?;
    let log_id = module
        .declare_function("gpum_log", Linkage::Import, &sig_un)
        .map_err(|e| e.to_string())?;
    let pow_id = module
        .declare_function("gpum_pow", Linkage::Import, &sig_bin)
        .map_err(|e| e.to_string())?;

    let mut sig = module.make_signature();
    for _ in 0..4 {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::F64));
    let func_id = module
        .declare_function("loglik", Linkage::Local, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fb_ctx = FunctionBuilderContext::new();

    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let exp = module.declare_func_in_func(exp_id, b.func);
        let log = module.declare_func_in_func(log_id, b.func);
        let pow = module.declare_func_in_func(pow_id, b.func);

        let entry = b.create_block();
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.append_block_param(header, types::F64); // accumulator
        b.append_block_param(header, types::I64); // observation index
        b.append_block_param(body, types::F64);
        b.append_block_param(body, types::I64);
        b.append_block_param(exit, types::F64);

        b.switch_to_block(entry);
        b.seal_block(entry);
        let p_params = b.block_params(entry)[0];
        let p_data = b.block_params(entry)[1];
        let n_obs = b.block_params(entry)[2];
        let n_cols = b.block_params(entry)[3];
        let zero_acc = b.ins().f64const(0.0f64);
        let zero_i = b.ins().iconst(types::I64, 0);
        b.ins().jump(header, &[zero_acc.into(), zero_i.into()]);

        b.switch_to_block(header);
        let acc = b.block_params(header)[0];
        let idx = b.block_params(header)[1];
        let cond = b.ins().icmp(IntCC::SignedLessThan, idx, n_obs);
        b.ins().brif(cond, body, &[acc.into(), idx.into()], exit,
                     &[acc.into()]);

        b.switch_to_block(body);
        let acc_b = b.block_params(body)[0];
        let idx_b = b.block_params(body)[1];
        // Base address of this observation's row: p_data + idx * n_cols * 8.
        let row_elems = b.ins().imul(idx_b, n_cols);
        let row_bytes = b.ins().imul_imm(row_elems, 8);
        let row_addr = b.ins().iadd(p_data, row_bytes);

        let mut stack: Vec<Value> = Vec::new();
        let n_instr = code.len() / 2;
        for pc in 0..n_instr {
            let op = code[2 * pc];
            let arg = code[2 * pc + 1] as usize;
            match op {
                0 => {
                    stack.push(b.ins().f64const(consts[arg]));
                }
                1 => {
                    let v = b.ins().load(types::F64, MemFlags::trusted(),
                                         p_params, (arg * 8) as i32);
                    stack.push(v);
                }
                2 => {
                    let v = b.ins().load(types::F64, MemFlags::trusted(),
                                         row_addr, (arg * 8) as i32);
                    stack.push(v);
                }
                3 => {
                    let r = stack.pop().unwrap();
                    let l = stack.pop().unwrap();
                    stack.push(b.ins().fadd(l, r));
                }
                4 => {
                    let r = stack.pop().unwrap();
                    let l = stack.pop().unwrap();
                    stack.push(b.ins().fsub(l, r));
                }
                5 => {
                    let r = stack.pop().unwrap();
                    let l = stack.pop().unwrap();
                    stack.push(b.ins().fmul(l, r));
                }
                6 => {
                    let r = stack.pop().unwrap();
                    let l = stack.pop().unwrap();
                    stack.push(b.ins().fdiv(l, r));
                }
                7 => {
                    let v = stack.pop().unwrap();
                    stack.push(b.ins().fneg(v));
                }
                8 => {
                    let v = stack.pop().unwrap();
                    let call = b.ins().call(exp, &[v]);
                    stack.push(b.inst_results(call)[0]);
                }
                9 => {
                    let v = stack.pop().unwrap();
                    let call = b.ins().call(log, &[v]);
                    stack.push(b.inst_results(call)[0]);
                }
                10 => {
                    let v = stack.pop().unwrap();
                    stack.push(b.ins().sqrt(v));
                }
                _ => {
                    let e = stack.pop().unwrap();
                    let base = stack.pop().unwrap();
                    let call = b.ins().call(pow, &[base, e]);
                    stack.push(b.inst_results(call)[0]);
                }
            }
        }
        let contrib = if n_instr > 0 {
            stack.pop().unwrap()
        } else {
            b.ins().f64const(0.0f64)
        };
        let acc_new = b.ins().fadd(acc_b, contrib);
        let idx_new = b.ins().iadd_imm(idx_b, 1);
        b.ins().jump(header, &[acc_new.into(), idx_new.into()]);

        b.seal_block(header);
        b.seal_block(body);

        b.switch_to_block(exit);
        b.seal_block(exit);
        let result = b.block_params(exit)[0];
        b.ins().return_(&[result]);

        b.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| e.to_string())?;

    let ptr = module.get_finalized_function(func_id);
    // SAFETY: `ptr` is the compiled `loglik` with exactly this C ABI.
    let func: extern "C" fn(*const f64, *const f64, i64, i64) -> f64 =
        unsafe { core::mem::transmute(ptr) };

    Ok(JitLoglik { _module: module, func })
}
