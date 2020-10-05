#' Function to derive the RoCELL log-posterior on the ML manifold, where for
#' each pair of confounding coefficients we can find a unique ML solution.
#'
#' @param sc24 confounding coefficient from 2 to 4 (scale-free)
#' @param sc34 confounding coefficient from 3 to 4 (scale-free)
#' @param Sigma_hat sample covariance matrix
#' @param N number of samples
#' @param k_slab prior slab component precision
#' @param k_spike prior spike component precision
#'
#' @return log-posterior value
#' @export
#'
#' @examples
#' Sigma <- get_Sigma(1, 0.05, 1, 1, 1, 1, 1, 1)
#' log_posterior_ML_manifold(0, 0, Sigma, 1)
log_posterior_ML_manifold <- function(sc24, sc34, Sigma_hat, N, k_slab = 1, k_spike = 100) {
  
  # at the ML solution, Sigma == Sigma_hat
  par <- get_ML_parameters_scale_free(Sigma_hat, sc24, sc34)
  
  # Compute the hessian of the log-likelihood wrt scale-free parameters
  hess <- Rcpp_log_likelihood_hessian_scale_free(
    Sigma_hat = Sigma_hat, N = N, sb21 = par$sB[2, 1], sb31 = par$sB[3, 1], sb32 = par$sB[3, 2],
    sc24 = sc24, sc34 = sc34, v1 = par$V[1, 1], v2 = par$V[2, 2], v3 = par$V[3, 3]
  )
  
  #  Remove the confounding coefficients block and compute the determinant to get Hessian term
  hess_term <- - log(det(-hess[-c(4, 5), -c(4, 5)])) / 2
  
  # Prior term on all parameters (\tild{B}, V, \tilde{C})
  prior_term <- Rcpp_log_spike_and_slab_scale_free(par$sB[2, 1], 0.5, k_slab, k_spike) +
    Rcpp_log_spike_and_slab_scale_free(par$sB[3, 1], 0.5, k_slab, k_spike) +
    Rcpp_log_spike_and_slab_scale_free(par$sB[3, 2], 0.5, k_slab, k_spike) +
    (- log(par$V[1, 1]) - log(par$V[1, 1]) - log(par$V[1, 1])) +
    stats::dnorm(sc24, log = TRUE) + stats::dnorm(sc34, log = TRUE)
  
  hess_term + prior_term
}


#' Function to derive the RoCELL log-determinant of the (negative) Hessian on 
#' the ML manifold, where for each pair of confounding coefficients we can find 
#' a unique ML solution.
#'
#' @param sc24 confounding coefficient from 2 to 4 (scale-free)
#' @param sc34 confounding coefficient from 3 to 4 (scale-free)
#' @param Sigma_hat sample covariance matrix
#' @param N number of samples
#'
#' @return the (negative) Hessian log-determinant
#' @export
#'
#' @examples
#' Sigma <- get_Sigma(1, 0.05, 1, 1, 1, 1, 1, 1)
#' log_hessian_term_ML_manifold(0, 0, Sigma, 1)
log_hessian_term_ML_manifold <- function(sc24, sc34, Sigma_hat, N) {
  
  # at the ML solution, Sigma == Sigma_hat
  par <- get_ML_parameters_scale_free_confounders(Sigma_hat, sc24, sc34)
  
  # Compute the hessian of the log-likelihood wrt scale-free confounders 
  # and the original structural parameters
  hess <- Rcpp_log_likelihood_hessian_scale_free_confounders(
    Sigma_hat = Sigma_hat, N = N, b21 = par$B[2, 1], b31 = par$B[3, 1], b32 = par$B[3, 2],
    sc24 = sc24, sc34 = sc34, v1 = par$V[1, 1], v2 = par$V[2, 2], v3 = par$V[3, 3]
  )
  
  # Remove the confounding coefficients block and compute the determinant
  det_neg_hess <- det(-hess[-c(4, 5), -c(4, 5)])
  
  - log(det_neg_hess) / 2
}




