#' Derive covariance matrix Sigma from model parameters
#'
#' @description The function covariance matrix for the linear-Gaussian model 
#' given all the parameter values.
#'
#' @param b21 structural coefficient for 1 -> 2
#' @param b31 structural coefficient for 1 -> 3
#' @param b32 structural coefficient for 2 -> 3
#' @param c24 confounding coefficient for 4 -> 2
#' @param c34 confounding coefficient for 4 -> 3
#' @param v1 intrinsic variance of 1
#' @param v2 intrinsic variance of 2
#' @param v3 intrinsic variance of 3
#'
#' @return model covariance matrix given the path coefficients and variances
#' @export
#'
#' @examples get_Sigma(1, 0.05, 1, 0, 0, 1, 1, 1)
get_Sigma <- function(b21, b31, b32, c24, c34, v1, v2, v3) {
  
  IIB <- matrix(c(1, b21, b31 + b32 * b21, 0, 1, b32, 0, 0, 1), 3, 3)
  C <- matrix(c(0, 0, 0, 0, 0, 0, 0, c24, c34), 3, 3)
  V <- diag(c(v1, v2, v3), 3)
  
  IIB %*% (V + C %*% t(C)) %*% t(IIB)
}


#' Derive scale-free structural parameters at ML solution given scale-free confounding coefficients
#'
#' @param Sigma_hat sample covariance matrix
#' @param sc24 scaled confounding coefficient for 4 -> 2
#' @param sc34 scaled confounding coefficient for 4 -> 3
#'
#' @return List containing matrix sB (scaled/dimensionless structural parameters)
#' and diagonal matrix V (intrinsic variances).
#' @export
#'
#' @examples
#' Sigma <- get_Sigma(1, 0.05, 1, 0, 0, 2, 3, 4)
#' get_ML_parameters_scale_free(Sigma, 0, 0)
get_ML_parameters_scale_free <- function(Sigma_hat, sc24, sc34) {
  
  spar <- Rcpp_get_params_scale_free(Sigma_hat, sc24, sc34)
  
  sB <- V <- diag(0, 3, 3)
  sB[lower.tri(sB)] <- spar[1:3]
  diag(V) <- spar[4:6]
  
  list(sB = sB, V = V)
  
  # sC <- matrix(c(0, 0, 0, 0, 0, 0, 0, sc24, sc34), 3, 3)
  # 
  # n <- nrow(Sigma_hat) # number of variables
  # Q <- t(chol(Sigma_hat)) # lower Cholesky factor of Sigma
  # 
  # # lower Cholesky factor of I + \tilde{C} \tilde{C}^T
  # L <- t(chol(diag(n) + sC %*% t(sC))) # lower Cholesky factor of
  # R <- Q %*% solve(L)
  # 
  # # square root of V
  # sqV <- diag(R)
  # 
  # # scale-free B matrix (\tilde{B})
  # sB <- diag(n) - L %*% solve(Q) %*% diag(sqV)
  # # intrinsic variance matrix
  # V <- diag(sqV * sqV)
  # 
  # list(sB = sB, V = V)
}

#' Derive structural parameters at ML solution given scale-free confounders.
#'
#' @param Sigma_hat sample covariance matrix
#' @param sc24 scaled confounding coefficient for 4 -> 2
#' @param sc34 scaled confounding coefficient for 4 -> 3
#'
#' @return List containing matrix B (structural parameters in original scale)
#' and diagonal matrix V (intrinsic variances).
#' @export
#'
#' @examples
#' Sigma <- get_Sigma(1, 0.05, 1, 0, 0, 2, 3, 4)
#' get_ML_parameters_scale_free_confounders(Sigma, 0, 0)
get_ML_parameters_scale_free_confounders <- function(Sigma_hat, sc24, sc34) {
  
  params <- get_ML_parameters_scale_free(Sigma_hat, sc24, sc34)
  
  sqV <- sqrt(params$V)
  
  # sC <- matrix(c(0, 0, 0, 0, 0, 0, 0, sc24, sc34), 3, 3)
  # 
  # n <- nrow(Sigma_hat) # number of variables
  # Q <- t(chol(Sigma_hat)) # lower Cholesky factor of Sigma_hat
  # 
  # # lower Cholesky factor of I + \tilde{C} \tilde{C}^T
  # L <- t(chol(diag(n) + sC %*% t(sC))) # lower Cholesky factor of
  # R <- Q %*% solve(L)
  # 
  # # square root of V
  # sqV <- diag(R)
  # 
  # # scale-free B matrix (\tilde{B})
  # B <- diag(n) - diag(sqV) %*% L %*% solve(Q) 
  # # intrinsic variance matrix
  # V <- diag(sqV * sqV)
  
  list(B = sqV %*% params$sB %*% solve(sqV), V = params$V)
}

#' Read RoCELL MultiNest output given sample covariance matrix used as input.
#'
#' @param filename character; Name of file containing RoCELL MultiNest output (.txt).
#' @param Sigma_hat sample covariance matrix
#'
#' @return coda::mcmc structure containing the RoCELL posterior samples
#' @export
#'
#' @examples
#' \dontrun{
#' Sigma <- get_Sigma(1, 0.05, 1, 1, 1, 1, 1, 1)
#' run_system_command(sprintf(
#'   "./RoCELL -s11 %g -s12 %g -s13 %g -s22 %g -s23 %g -s33 %g",
#'   Sigma[1, 1], Sigma[1, 2], Sigma[1, 3], Sigma[2, 2], Sigma[2, 3], Sigma[3, 3]
#' ))
#' read_MultiNest_samples("output_RoCELL-.txt", Sigma)
#' }
read_MultiNest_samples <- function(filename, Sigma_hat) {
  
  samples <- utils::read.table(filename)
  
  weights <- samples[, 1]
  
  mcmc_B <- sapply(samples[, c(3, 4)], function(sc) stats::qnorm(sc))
  
  mcmc_A <- t(apply(mcmc_B, 1, function(sc) {
    params <- get_ML_parameters_scale_free_confounders(Sigma_hat, sc[1], sc[2])
    c(params$B[3, 1], params$B[3, 2], params$V[2, 2], params$V[3, 3])
  }))
  
  mcmc_samples <- coda::mcmc(cbind(mcmc_A, mcmc_B, samples[, 1]))
  colnames(mcmc_samples) <- c('b31', 'b32', 'v2', 'v3', 'sc24', 'sc34', 'w')
  
  mcmc_samples
}

#' Read log evidence from MultiNest stats output file
#'
#' @param filename character; Name of MultiNest stats.dat output file.
#'
#' @return Numeric vector containing the estimated log-evidence as the first
#' element and its estimated standard error as the second element.
#' @export
#'
#' @examples
#' \dontrun{
#' Sigma <- get_Sigma(1, 0.05, 1, 1, 1, 1, 1, 1)
#' run_system_command(sprintf(
#'   "./RoCELL -s11 %g -s12 %g -s13 %g -s22 %g -s23 %g -s33 %g",
#'   Sigma[1, 1], Sigma[1, 2], Sigma[1, 3], Sigma[2, 2], Sigma[2, 3], Sigma[3, 3]
#' ))
#' read_MultiNest_log_evidence("output_RoCELL-stats.dat")
#' }
read_MultiNest_log_evidence <- function(filename = "output_RoCELL-stats.dat") {
  first_line <- readLines(filename, n = 1)
  sapply(stringr::str_extract_all(first_line, "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)", simplify = TRUE), as.numeric)
}


#' A more portable function for invoking a system command
#'
#' @param command String containing the system command to be invoked.
#' @param Windows_shell Full path to the Windows shell to be used for invoking 
#' the system command. MSYS2+MinGW has been tested and is recommended.
#' @param ... Further shell parameters.
#'
#' @return See \link[base]{system} for Linux and \link[pkg]{shell} for Windows.
#' @export
#'
#' @examples run_system_command("echo Hello!")
run_system_command <- function(command, Windows_shell = 'C:/msys64/msys2_shell.cmd -defterm -here -no-start -mingw64', ...) {
  
  if(.Platform$OS.type == "windows") {
    shell(shQuote(command), Windows_shell, flag = "-c", ...)
  } else {
    system(command)
  }
}
  