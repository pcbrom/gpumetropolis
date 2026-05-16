# Expression compiler for the gpumetropolis DSL.
#
# Translates a log-density expression, declared by the user as a one-sided
# formula, into the flat stack-machine bytecode that the CubeCL interpreter
# kernel executes. The opcodes here are kept in sync with src/rust/src/interp.rs.
#
# Supported operations: + - * / on two operands, unary -, ^, and the unary
# functions exp, log (natural) and sqrt. Any other symbol or function is
# rejected with a clear error, so an unsupported model fails at compile time,
# not silently.

# Opcodes, matching the interpreter kernel.
.gpum_op <- c(
  PUSH_CONST = 0L, PUSH_PARAM = 1L, PUSH_DATA = 2L,
  ADD = 3L, SUB = 4L, MUL = 5L, DIV = 6L, NEG = 7L,
  EXP = 8L, LOG = 9L, SQRT = 10L, POW = 11L
)

# Maximum operand-stack depth of the interpreter kernel (STACK_SIZE).
.gpum_stack_size <- 32L

# Compile one expression to bytecode.
#
# `expr` is a call, symbol or numeric (the right-hand side of the loglik
# formula). `params` and `data` are the declared parameter and data-column
# names. Returns a list with the integer `code` (opcode/arg pairs), the numeric
# `consts` pool and the peak stack `depth`.
.gpum_compile <- function(expr, params, data) {
  consts <- numeric(0)
  code <- integer(0)
  depth <- 0L
  max_depth <- 0L

  push_depth <- function(delta) {
    depth <<- depth + delta
    if (depth > max_depth) max_depth <<- depth
  }
  emit_op <- function(op, arg = 0L) {
    code <<- c(code, op, as.integer(arg))
  }
  const_index <- function(v) {
    hit <- which(consts == v)
    if (length(hit)) return(hit[1L] - 1L)
    consts <<- c(consts, v)
    length(consts) - 1L
  }

  emit <- function(node) {
    if (is.numeric(node) && length(node) == 1L) {
      emit_op(.gpum_op["PUSH_CONST"], const_index(as.numeric(node)))
      push_depth(1L)
    } else if (is.symbol(node)) {
      nm <- as.character(node)
      p_idx <- match(nm, params)
      d_idx <- match(nm, data)
      if (!is.na(p_idx)) {
        emit_op(.gpum_op["PUSH_PARAM"], p_idx - 1L)
      } else if (!is.na(d_idx)) {
        emit_op(.gpum_op["PUSH_DATA"], d_idx - 1L)
      } else {
        stop("unknown symbol in the log-density: '", nm,
             "'. Declare it in `params` or `data`.", call. = FALSE)
      }
      push_depth(1L)
    } else if (is.call(node)) {
      fn <- as.character(node[[1L]])
      args <- as.list(node)[-1L]
      binary <- function(op) {
        emit(args[[1L]])
        emit(args[[2L]])
        emit_op(op)
        push_depth(-1L)
      }
      unary_fn <- function(op) {
        if (length(args) != 1L) {
          stop("`", fn, "` takes one argument in the log-density.",
               call. = FALSE)
        }
        emit(args[[1L]])
        emit_op(op)
      }
      if (fn == "(") {
        emit(args[[1L]])
      } else if (fn == "+" && length(args) == 2L) {
        binary(.gpum_op["ADD"])
      } else if (fn == "-" && length(args) == 2L) {
        binary(.gpum_op["SUB"])
      } else if (fn == "-" && length(args) == 1L) {
        emit(args[[1L]])
        emit_op(.gpum_op["NEG"])
      } else if (fn == "*" && length(args) == 2L) {
        binary(.gpum_op["MUL"])
      } else if (fn == "/" && length(args) == 2L) {
        binary(.gpum_op["DIV"])
      } else if (fn == "^" && length(args) == 2L) {
        binary(.gpum_op["POW"])
      } else if (fn == "exp") {
        unary_fn(.gpum_op["EXP"])
      } else if (fn == "log") {
        unary_fn(.gpum_op["LOG"])
      } else if (fn == "sqrt") {
        unary_fn(.gpum_op["SQRT"])
      } else {
        stop("unsupported function in the log-density: '", fn,
             "'. Supported: + - * / ^ exp log sqrt.", call. = FALSE)
      }
    } else {
      stop("unsupported term in the log-density expression.", call. = FALSE)
    }
  }

  emit(expr)
  if (max_depth > .gpum_stack_size) {
    stop("the log-density expression is too deep for the interpreter ",
         "(needs ", max_depth, " stack slots, limit ", .gpum_stack_size,
         ").", call. = FALSE)
  }
  list(code = as.integer(code), consts = as.numeric(consts),
       depth = max_depth)
}

# Extract the right-hand side expression of a one-sided formula, or accept a
# bare quoted expression.
.gpum_rhs <- function(x) {
  if (inherits(x, "formula")) {
    return(x[[length(x)]])
  }
  if (is.call(x) || is.symbol(x) || is.numeric(x)) {
    return(x)
  }
  stop("expected a one-sided formula such as `~ expr`.", call. = FALSE)
}
