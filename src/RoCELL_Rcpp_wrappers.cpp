#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

#define CALL_ROCELL_FROM_R

#include "../posterior/RoCELL.h"


//' Compute log-value of spike-and-slab prior for parameter given mixture weight.
//'
//' @param x numeric value of random variable
//' @param w mixture weight between 0 and 1, with 1 indicating slab and 0 spike
//' @param k_slab slab component precision
//' @param k_spike spike component precision
//' 
//' @return Log-value of spike-and-slab density computed at x. 
// [[Rcpp::export]]
double Rcpp_log_spike_and_slab_scale_free(
  double x, double w, double k_slab = 1, double k_spike = 100) {
  
  return log_spike_and_slab_scale_free(x, w, k_slab, k_spike);
}

//' Derive scale-free structural parameters at ML solution given scale-free confounding coefficients.
//'
//' @param Sigma_hat sample covariance matrix
//' @param sc24 scaled confounding coefficient for 4 -> 2
//' @param sc34 scaled confounding coefficient for 4 -> 3
//'
//' @return Vector containing scaled structural coefficients (sb21, sb31, sb32)
//' and intrinsic variances (v1, v2, v3), in this order.
// [[Rcpp::export]]
arma::vec Rcpp_get_params_scale_free(arma::mat Sigma_hat, double sc24, double sc34) {
  
  return get_params_scale_free(Sigma_hat, sc24, sc34);
}

//' Compute RoCELL log-posterior given sample covariance matrix and parameters.
//'
//' @param Sigma_hat sample covariance matrix
//' @param N number of samples
//' @param sb21 scaled structural coefficient for 1 -> 2
//' @param sb31 scaled structural coefficient for 1 -> 3
//' @param sb32 scaled structural coefficient for 1 -> 3
//' @param sc24 scaled confounding coefficient for 4 -> 2
//' @param sc34 scaled confounding coefficient for 4 -> 3
//' @param v1 intrinsic variance of 1
//' @param v2 intrinsic variance of 2
//' @param v3 intrinsic variance of 3
//' @param slab_precision slab component precision
//' @param spike_precision spike component precision
//' @param w32 Mixture weight for spike-and-slab prior applied to sb32.
//'
//' @return RoCELL log-posterior computed at the specified parameter values
// [[Rcpp::export]]
double Rcpp_log_posterior_RoCELL(
    arma::mat Sigma_hat, unsigned N, double sb21, double sb31, double sb32,
    double sc24, double sc34, double v1, double v2, double v3,
    double slab_precision, double spike_precision, double w32 = 0.5) {
  
  return log_posterior_RoCELL(Sigma_hat, N, sb21, sb31, sb32, sc24, sc34,
                              v1, v2, v3, slab_precision, spike_precision, w32);
}

//' Compute log-likelihood gradient wrt scale-free parameters.
//'
//' @param Sigma_hat sample covariance matrix
//' @param N number of samples
//' @param sb21 scaled structural coefficient for 1 -> 2
//' @param sb31 scaled structural coefficient for 1 -> 3
//' @param sb32 scaled structural coefficient for 1 -> 3
//' @param sc24 scaled confounding coefficient for 4 -> 2
//' @param sc34 scaled confounding coefficient for 4 -> 3
//' @param v1 intrinsic variance of 1
//' @param v2 intrinsic variance of 2
//' @param v3 intrinsic variance of 3
//'
//' @return log-likelihood gradient vector of length 8 with the parameters ordered as above
// [[Rcpp::export]]
arma::vec Rcpp_log_likelihood_gradient_scale_free(
    arma::mat Sigma_hat, unsigned N, double sb21, double sb31, double sb32,
    double sc24, double sc34, double v1, double v2, double v3) {
  
  return log_likelihood_gradient_scale_free(Sigma_hat, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3);
}

//' Compute log-likelihood Hessian wrt scale-free parameters.
//'
//' @param Sigma_hat sample covariance matrix
//' @param N number of samples
//' @param sb21 scaled structural coefficient for 1 -> 2
//' @param sb31 scaled structural coefficient for 1 -> 3
//' @param sb32 scaled structural coefficient for 1 -> 3
//' @param sc24 scaled confounding coefficient for 4 -> 2
//' @param sc34 scaled confounding coefficient for 4 -> 3
//' @param v1 intrinsic variance of 1
//' @param v2 intrinsic variance of 2
//' @param v3 intrinsic variance of 3
//'
//' @return log-likelihood Hessian matrix (8x8) with the parameters ordered as above
// [[Rcpp::export]]
arma::mat Rcpp_log_likelihood_hessian_scale_free(
    arma::mat Sigma_hat, unsigned N, double sb21, double sb31, double sb32,
    double sc24, double sc34, double v1, double v2, double v3) {
  return log_likelihood_hessian_scale_free(Sigma_hat, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3);
}

//' Compute log-likelihood hessian wrt scaled confounding coefficients and
//' structural coefficients in original scale (unscaled).
//'
//' @param Sigma_hat sample covariance matrix
//' @param N number of samples
//' @param b21 structural coefficient for 1 -> 2
//' @param b31 structural coefficient for 1 -> 3
//' @param b32 structural coefficient for 1 -> 3
//' @param sc24 scaled confounding coefficient for 4 -> 2
//' @param sc34 scaled confounding coefficient for 4 -> 3
//' @param v1 intrinsic variance of 1
//' @param v2 intrinsic variance of 2
//' @param v3 intrinsic variance of 3
//'
//' @return log-likelihood Hessian matrix (8x8) with the parameters ordered as above
// [[Rcpp::export]]
arma::mat Rcpp_log_likelihood_hessian_scale_free_confounders(
    arma::mat Sigma_hat, unsigned N, double b21, double b31, double b32,
    double sc24, double sc34, double v1, double v2, double v3) {
  return log_likelihood_hessian_scale_free_confounders(Sigma_hat, N, b21, b31, b32, sc24, sc34, v1, v2, v3);
}
