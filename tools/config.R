# Note: Any variables prefixed with `.` are used for text
# replacement in the Makevars.in and Makevars.win.in

# check the packages MSRV first
source("tools/msrv.R")

# check DEBUG and NOT_CRAN environment variables
env_debug <- Sys.getenv("DEBUG")
env_not_cran <- Sys.getenv("NOT_CRAN")

# check if the vendored zip file exists
vendor_exists <- file.exists("src/rust/vendor.tar.xz")

is_not_cran <- env_not_cran != ""
is_debug <- env_debug != ""

if (is_debug) {
  # if we have DEBUG then we set not cran to true
  # CRAN is always release build
  is_not_cran <- TRUE
  message("Creating DEBUG build.")
}

if (!is_not_cran) {
  message("Building for CRAN.")
}

# we set cran flags only if NOT_CRAN is empty and if
# the vendored crates are present.
.cran_flags <- ifelse(
  !is_not_cran && vendor_exists,
  "-j 2 --offline",
  ""
)

# when DEBUG env var is present we use `--debug` build
.profile <- ifelse(is_debug, "", "--release")
.clean_targets <- ifelse(is_debug, "", "$(TARGET_DIR)")

# We specify this target when building for webR
webr_target <- "wasm32-unknown-emscripten"

# here we check if the platform we are building for is webr
is_wasm <- identical(R.version$platform, webr_target)

# print to terminal to inform we are building for webr
if (is_wasm) {
  message("Building for WebR")
}

# we check if we are making a debug build or not
# if so, the LIBDIR environment variable becomes:
# LIBDIR = $(TARGET_DIR)/{wasm32-unknown-emscripten}/debug
# this will be used to fill out the LIBDIR env var for Makevars.in
target_libpath <- if (is_wasm) "wasm32-unknown-emscripten" else NULL
cfg <- if (is_debug) "debug" else "release"

# used to replace @LIBDIR@
.libdir <- paste(c(target_libpath, cfg), collapse = "/")

# use this to replace @TARGET@
# we specify the target _only_ on webR
# there may be use cases later where this can be adapted or expanded
.target <- ifelse(is_wasm, paste0("--target=", webr_target), "")

# add panic exports only for WASM builds
.panic_exports <- ifelse(
  is_wasm,
  "CARGO_PROFILE_DEV_PANIC=\"abort\" CARGO_PROFILE_RELEASE_PANIC=\"abort\" ",
  ""
)

# Detect GPU backends available at install time. The package builds CPU by
# default and adds Cargo features for any GPU backend whose toolchain is
# present on PATH, so a source install on a machine with CUDA or Vulkan
# tooling produces a binary that exposes the matching backends without the
# user passing any flag. Optional env-var overrides for CI or diagnosis:
#   GPUMETROPOLIS_BACKENDS = auto|cpu|cuda|vulkan|cuda,vulkan
#     auto (or unset) runs detection; any other value forces the set.
#   GPUMETROPOLIS_CUDA   = 0|1   forces a single backend after auto.
#   GPUMETROPOLIS_VULKAN = 0|1
parse_backends_spec <- function(spec) {
  parts <- unlist(strsplit(spec, "[, ]+"))
  parts <- tolower(trimws(parts[nzchar(parts)]))
  unique(parts[parts != "cpu"])
}

force_backend_flag <- function(current, value, name) {
  if (!nzchar(value)) return(current)
  v <- tolower(value)
  if (v %in% c("0", "false", "no",  "off")) return(setdiff(current, name))
  if (v %in% c("1", "true",  "yes", "on"))  return(union(current,   name))
  current
}

env_backends <- Sys.getenv("GPUMETROPOLIS_BACKENDS", unset = "")
env_cuda     <- Sys.getenv("GPUMETROPOLIS_CUDA",     unset = "")
env_vulkan   <- Sys.getenv("GPUMETROPOLIS_VULKAN",   unset = "")

if (nzchar(env_backends) && tolower(env_backends) != "auto") {
  selected_backends <- parse_backends_spec(env_backends)
} else {
  selected_backends <- character()
  if (nzchar(Sys.which("nvcc")))       selected_backends <- c(selected_backends, "cuda")
  if (nzchar(Sys.which("vulkaninfo"))) selected_backends <- c(selected_backends, "vulkan")
}

selected_backends <- force_backend_flag(selected_backends, env_cuda,   "cuda")
selected_backends <- force_backend_flag(selected_backends, env_vulkan, "vulkan")

# WebAssembly target cannot pull in the CUDA or Vulkan crates.
if (is_wasm) selected_backends <- character()

selected_backends <- intersect(selected_backends, c("cuda", "vulkan"))

overridden <- nzchar(env_backends) || nzchar(env_cuda) || nzchar(env_vulkan)

if (length(selected_backends) > 0) {
  .cargo_features <- paste("--features", paste(selected_backends, collapse = ","))
  message("gpumetropolis: building backends = ",
          paste(c("cpu", selected_backends), collapse = ", "), ".")
} else if (overridden) {
  .cargo_features <- ""
  message("gpumetropolis: building CPU only (override).")
} else {
  .cargo_features <- ""
  message("gpumetropolis: building CPU only (no GPU toolchain detected). ",
          "Set GPUMETROPOLIS_BACKENDS to override.")
}

# read in the Makevars.in file checking
is_windows <- .Platform[["OS.type"]] == "windows"

# if windows we replace in the Makevars.win.in
mv_fp <- ifelse(
  is_windows,
  "src/Makevars.win.in",
  "src/Makevars.in"
)

# set the output file
mv_ofp <- ifelse(
  is_windows,
  "src/Makevars.win",
  "src/Makevars"
)

# delete the existing Makevars{.win/.wasm}
if (file.exists(mv_ofp)) {
  message("Cleaning previous `", mv_ofp, "`.")
  invisible(file.remove(mv_ofp))
}

# read as a single string
mv_txt <- readLines(mv_fp)

# replace placeholder values
new_txt <- gsub("@CRAN_FLAGS@", .cran_flags, mv_txt) |>
  gsub("@CARGO_FEATURES@", .cargo_features, x = _) |>
  gsub("@PROFILE@", .profile, x = _) |>
  gsub("@CLEAN_TARGET@", .clean_targets, x = _) |>
  gsub("@LIBDIR@", .libdir, x = _) |>
  gsub("@TARGET@", .target, x = _) |>
  gsub("@PANIC_EXPORTS@", .panic_exports, x = _)

message("Writing `", mv_ofp, "`.")
con <- file(mv_ofp, open = "wb")
writeLines(new_txt, con, sep = "\n")
close(con)

message("`tools/config.R` has finished.")
