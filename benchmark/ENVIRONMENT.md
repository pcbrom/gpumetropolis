# Benchmark environment

Machine and software snapshot of the host that ran the registered
experiment of EXPERIMENT_PROTOCOL.md. Regenerate with
`bash benchmark/capture_env.sh`.

Captured: 2026-05-16 17:27:48 UTC

## Operating system
```
Ubuntu 24.04.4 LTS
kernel: Linux 6.17.0-23-generic
arch  : x86_64
```

## CPU
```
Architecture: x86_64
CPU(s): 24
Model name: AMD Ryzen 9 9900X3D 12-Core Processor
Thread(s) per core: 2
Core(s) per socket: 12
Socket(s): 1
CPU max MHz: 5575.5732
CPU min MHz: 604.5810
L3 cache: 128 MiB (2 instances)
frequency governor: powersave
```

## Memory
```
 total used free shared buff/cache available
Mem: 60Gi 17Gi 28Gi 419Mi 14Gi 42Gi
Swap: 953Gi 620Ki 953Gi
```

## GPU
```
01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)
10:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 13c0 (rev ca)
--- NVIDIA ---
NVIDIA GeForce RTX 4090, 535.288.01, 24564 MiB
```

## Toolchain
```
rustc 1.95.0 (59807616e 2026-04-14)
cargo 1.95.0 (f2d3ce0bd 2026-03-21)
R version 4.6.0 (2026-04-24) -- "Because it was There"
```

## R packages
```
gpumetropolis  0.0.0.9000
MCMCpack       1.7.1
mcmc           0.9.8
nimble         1.4.2
BayesianTools  0.1.9
coda           0.19.4.1
cmdstanr       0.9.0
rextendr       0.5.0
cmdstan        2.38.0
```
