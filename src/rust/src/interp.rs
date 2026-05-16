//! Bytecode interpreter kernel: a generic per-observation log-likelihood.
//!
//! The user's log-likelihood expression is compiled, on the R side, to a flat
//! stack-machine bytecode. One fixed CubeCL kernel interprets that bytecode, so
//! any expression in the supported operation set runs on the CPU and GPU
//! runtimes from this single source, with no runtime code generation.
//!
//! Opcodes (kept in sync with the R-side compiler in `R/dsl.R`):
//!   0 PUSH_CONST(arg = constant index)
//!   1 PUSH_PARAM(arg = parameter index)
//!   2 PUSH_DATA(arg = data column index)
//!   3 ADD   4 SUB   5 MUL   6 DIV   7 NEG
//!   8 EXP   9 LOG   10 SQRT   11 POW
//!
//! Parallelism: one thread per chain. Each thread runs the virtual machine for
//! every observation and accumulates the per-observation contributions. Every
//! thread executes the same bytecode, so there is no warp divergence.

use cubecl::prelude::*;

const STACK_SIZE: usize = 32;

#[cube(launch_unchecked)]
fn loglik_sum_kernel(
    code: &Array<u32>,     // 2 entries per instruction: opcode, arg
    consts: &Array<f32>,   // constant pool
    params: &Array<f32>,   // n_chains * n_params, grouped by chain
    data: &Array<f32>,     // n_obs * n_cols, grouped by observation
    meta: &Array<u32>,     // [n_instr, n_params, n_cols, n_obs]
    out: &mut Array<f32>,  // n_chains, one log-likelihood sum per chain
) {
    let chain = ABSOLUTE_POS;
    if chain < out.len() {
        let n_instr = usize::cast_from(meta[0]);
        let n_params = usize::cast_from(meta[1]);
        let n_cols = usize::cast_from(meta[2]);
        let n_obs = usize::cast_from(meta[3]);
        let param_base = chain * n_params;

        let mut acc = 0.0f32;
        let mut obs = 0usize;
        while obs < n_obs {
            let data_base = obs * n_cols;
            let mut stack = Array::<f32>::new(STACK_SIZE);
            let mut sp = 0usize;
            let mut pc = 0usize;
            while pc < n_instr {
                let op = code[2 * pc];
                let arg = usize::cast_from(code[2 * pc + 1]);
                if op == 0 {
                    stack[sp] = consts[arg];
                    sp += 1;
                } else if op == 1 {
                    stack[sp] = params[param_base + arg];
                    sp += 1;
                } else if op == 2 {
                    stack[sp] = data[data_base + arg];
                    sp += 1;
                } else if op == 3 {
                    stack[sp - 2] = stack[sp - 2] + stack[sp - 1];
                    sp -= 1;
                } else if op == 4 {
                    stack[sp - 2] = stack[sp - 2] - stack[sp - 1];
                    sp -= 1;
                } else if op == 5 {
                    stack[sp - 2] = stack[sp - 2] * stack[sp - 1];
                    sp -= 1;
                } else if op == 6 {
                    stack[sp - 2] = stack[sp - 2] / stack[sp - 1];
                    sp -= 1;
                } else if op == 7 {
                    stack[sp - 1] = -stack[sp - 1];
                } else if op == 8 {
                    stack[sp - 1] = stack[sp - 1].exp();
                } else if op == 9 {
                    stack[sp - 1] = stack[sp - 1].ln();
                } else if op == 10 {
                    stack[sp - 1] = stack[sp - 1].sqrt();
                } else {
                    stack[sp - 2] = stack[sp - 2].powf(stack[sp - 1]);
                    sp -= 1;
                }
                pc += 1;
            }
            acc += stack[0];
            obs += 1;
        }
        out[chain] = acc;
    }
}

/// One compiled log-likelihood program plus the data it runs over.
pub struct Program<'a> {
    pub code: &'a [u32],
    pub consts: &'a [f32],
    pub n_params: u32,
    pub data: &'a [f32],
    pub n_cols: u32,
    pub n_obs: u32,
}

fn run<R: Runtime>(prog: &Program, params: &[f32], n_chains: usize) -> Vec<f32> {
    let client = R::client(&Default::default());

    let code_h = client.create_from_slice(u32::as_bytes(prog.code));
    let consts_h = client.create_from_slice(f32::as_bytes(prog.consts));
    let params_h = client.create_from_slice(f32::as_bytes(params));
    let data_h = client.create_from_slice(f32::as_bytes(prog.data));
    let meta = [
        (prog.code.len() / 2) as u32,
        prog.n_params,
        prog.n_cols,
        prog.n_obs,
    ];
    let meta_h = client.create_from_slice(u32::as_bytes(&meta));
    let out_h = client.empty(n_chains * core::mem::size_of::<f32>());

    let block = 256u32;
    let count = CubeCount::Static((n_chains as u32).div_ceil(block), 1, 1);
    let dim = CubeDim { x: block, y: 1, z: 1 };

    unsafe {
        loglik_sum_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(code_h, prog.code.len()),
            ArrayArg::from_raw_parts(consts_h, prog.consts.len()),
            ArrayArg::from_raw_parts(params_h, params.len()),
            ArrayArg::from_raw_parts(data_h, prog.data.len()),
            ArrayArg::from_raw_parts(meta_h, meta.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n_chains),
        );
    }

    let bytes = client.read_one_unchecked(out_h);
    f32::from_bytes(&bytes).to_vec()
}

/// Evaluate the compiled log-likelihood sum for every chain on the chosen
/// backend. `params` holds `n_chains * prog.n_params` values grouped by chain.
pub fn loglik_sum(
    prog: &Program,
    params: &[f32],
    n_chains: usize,
    backend: crate::gpu::Backend,
) -> Vec<f32> {
    match backend {
        crate::gpu::Backend::Cpu => run::<cubecl::cpu::CpuRuntime>(prog, params, n_chains),
        crate::gpu::Backend::Cuda => run::<cubecl::cuda::CudaRuntime>(prog, params, n_chains),
    }
}
