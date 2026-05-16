#!/usr/bin/env bash
# Capture the full machine and software environment of the benchmark host,
# so a third party can reproduce or audit the registered run. Writes a
# markdown snapshot to benchmark/ENVIRONMENT.md. Re-runnable: run again to
# refresh the snapshot.
#
# Usage: bash benchmark/capture_env.sh

set -u
out="benchmark/ENVIRONMENT.md"

{
  echo "# Benchmark environment"
  echo
  echo "Machine and software snapshot of the host that ran the registered"
  echo "experiment of EXPERIMENT_PROTOCOL.md. Regenerate with"
  echo "\`bash benchmark/capture_env.sh\`."
  echo
  echo "Captured: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo

  echo "## Operating system"
  echo '```'
  (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo "unknown"
  echo "kernel: $(uname -sr)"
  echo "arch  : $(uname -m)"
  echo '```'
  echo

  echo "## CPU"
  echo '```'
  LC_ALL=C lscpu 2>/dev/null | grep -E "^(Model name|Architecture|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz|CPU min MHz|L3 cache):" \
    | sed 's/  */ /g'
  gov=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | paste -sd,)
  echo "frequency governor: ${gov:-unavailable}"
  echo '```'
  echo

  echo "## Memory"
  echo '```'
  LC_ALL=C free -h 2>/dev/null | sed 's/  */ /g'
  echo '```'
  echo

  echo "## GPU"
  echo '```'
  lspci 2>/dev/null | grep -iE "vga|3d|display" || echo "lspci unavailable"
  echo "--- NVIDIA ---"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi unavailable"
  echo '```'
  echo

  echo "## Toolchain"
  echo '```'
  command -v rustc >/dev/null 2>&1 && rustc --version || echo "rustc: absent"
  command -v cargo >/dev/null 2>&1 && cargo --version || echo "cargo: absent"
  R --version 2>/dev/null | head -1 || echo "R: absent"
  echo '```'
  echo

  echo "## R packages"
  echo '```'
  Rscript -e '
    pk <- c("gpumetropolis","MCMCpack","mcmc","nimble","BayesianTools",
            "coda","cmdstanr","rextendr")
    for (p in pk) cat(sprintf("%-14s %s\n", p,
      tryCatch(as.character(packageVersion(p)), error = function(e) "absent")))
    cat(sprintf("%-14s %s\n", "cmdstan",
      tryCatch(cmdstanr::cmdstan_version(), error = function(e) "absent")))
  ' 2>/dev/null || echo "Rscript unavailable"
  echo '```'
} > "$out"

echo "wrote $out"
