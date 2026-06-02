# Package hooks. The startup hook notifies the user, only in interactive
# sessions and only on success, when a newer version of gpumetropolis is
# available on the R-universe channel of the maintainer. The check is
# silent on any failure (offline, DNS, parse error) so a missing network
# never blocks library() or prints a confusing diagnostic.

.onAttach <- function(libname, pkgname) {
  if (!interactive()) return(invisible(NULL))
  if (nzchar(Sys.getenv("GPUMETROPOLIS_NO_VERSION_CHECK"))) {
    return(invisible(NULL))
  }
  current <- as.character(utils::packageVersion(pkgname))
  latest <- tryCatch(.gpum_latest_runiverse(), error = function(e) NULL)
  if (is.null(latest)) return(invisible(NULL))
  cmp <- tryCatch(utils::compareVersion(latest, current),
                  error = function(e) NA_integer_)
  if (is.na(cmp) || cmp <= 0L) return(invisible(NULL))
  packageStartupMessage(sprintf(
    "gpumetropolis %s available (installed: %s). Update from R-universe with:\n",
    latest, current
  ),
  "  install.packages(\"gpumetropolis\", type = \"source\",\n",
  "                   repos = c(\"https://pcbrom.r-universe.dev\",\n",
  "                             \"https://cloud.r-project.org\"))\n",
  "Set GPUMETROPOLIS_NO_VERSION_CHECK to silence."
  )
  invisible(NULL)
}

.gpum_latest_runiverse <- function(
    url = "https://pcbrom.r-universe.dev/api/packages/gpumetropolis",
    timeout = 2) {
  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = timeout)
  con <- base::url(url, open = "rb")
  on.exit(close(con), add = TRUE)
  raw <- readLines(con, warn = FALSE, encoding = "UTF-8")
  .gpum_parse_version(paste(raw, collapse = " "))
}

# Extract the "Version" field from an r-universe JSON payload without taking
# a runtime dependency on a JSON parser. Returns NULL when the field is
# absent or unparsable.
.gpum_parse_version <- function(text) {
  m <- regmatches(
    text,
    regexec('"Version"[[:space:]]*:[[:space:]]*"([^"]+)"', text)
  )[[1]]
  if (length(m) < 2L) return(NULL)
  m[[2L]]
}
