#' gpumetropolis: GPU-portable vendor-agnostic Metropolis-Hastings sampler
#'
#' Provides a batched random-walk Metropolis-Hastings sampler whose log-density
#' evaluation kernel is written to be portable across GPU vendors and CPU back
#' ends. This version ships the CPU reference sampler
#' ([metropolis_gaussian_mean()]) and the distributional equivalence harness
#' ([rhat()], [ess()], [ks_equivalence()]) against which later GPU versions are
#' checked.
#'
#' @keywords internal
"_PACKAGE"
