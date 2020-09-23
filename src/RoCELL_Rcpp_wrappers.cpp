#include <RcppArmadillo.h>
// #include <RcppEigen.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

#define CALL_ROCELL_FROM_R

#include "RoCELL.h"

// Eigen::MatrixXd Rcpp_two_times(Eigen::MatrixXd x) {
//   return two_times(x);
// }

// [[Rcpp::export]]
void Rcpp_print() {
  
#if defined(RcppArmadillo__RcppArmadillo__h)
  std::cout << "RcppArmadillo enabled" << std::endl;
#else
  std::cout << "RcppArmadillo not enabled" << std::endl;
#endif
  
  print();
  
  
}

// [[Rcpp::export]]
arma::mat Rcpp_two_times_arma(arma::mat x) {
  return two_times_arma(x);
}

// [[Rcpp::export]]
double Rcpp_log_posterior_RoCELL(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32,
    double sc24, double sc34, double v1, double v2, double v3,
    double slab_precision, double spike_precision, double w32 = 0.5) {
  
  return log_posterior_RoCELL(sncsm, N, sb21, sb31, sb32, sc24, sc34,
                              v1, v2, v3, slab_precision, spike_precision, w32);
}

// [[Rcpp::export]]
arma::vec Rcpp_log_likelihood_gradient_scale_free(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32,
    double sc24, double sc34, double v1, double v2, double v3) {
  
  return log_likelihood_gradient_scale_free(sncsm, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3);
}

// [[Rcpp::export]]
arma::mat Rcpp_log_likelihood_hessian_scale_free(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32,
    double sc24, double sc34, double v1, double v2, double v3) {
  return log_likelihood_hessian_scale_free(sncsm, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3);
}

// [[Rcpp::export]]
arma::mat Rcpp_log_likelihood_hessian_scale_free_confounders(
    arma::mat sncsm, unsigned N, double b21, double b31, double b32,
    double sc24, double sc34, double v1, double v2, double v3) {
  return log_likelihood_hessian_scale_free_confounders(sncsm, N, b21, b31, b32, sc24, sc34, v1, v2, v3);
}
