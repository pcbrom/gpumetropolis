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
