#ifndef ROCELL_H
#define ROCELL_H

#include <armadillo>
// #include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Cholesky>

//' Compute 
//'
//' @author Ioan Gabriel Bucur
arma::vec log_likelihood_gradient_scale_free(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32, 
    double sc24, double sc34, double v1, double v2, double v3);

//' Difference between two given timezones at a specified date.
//'
//' Time zone offsets vary by date, and this helper function computes
//' the difference (in hours) between two time zones for a given date time.
//'
//' @param sncsm Scaled non-centered scatter matrix.
//' @param N Number of observations.
//'
//' @return A numeric value with the difference (in hours) between the first and
//' second time zone at the given date
//' 
//' @author Ioan Gabriel Bucur
//' 
//' @examples
arma::mat log_likelihood_hessian_scale_free(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32, 
    double sc24, double sc34, double v1, double v2, double v3);

arma::mat log_likelihood_hessian_scale_free_confounders(
    arma::mat sncsm, unsigned N, double b21, double b31, double b32,
    double sc24, double sc34, double v1, double v2, double v3);

inline double log_spike_and_slab(double alpha, double vtail, double varrow, double beta = 0.5, double k1 = 1, double k2 = 1e2) {
  
  double var1 = varrow / vtail / k1;
  double var2 = varrow / vtail / k2;
  
  if (fabs(beta) < 1e-6) return - 0.5 * (alpha * alpha / var2 + log(2 * M_PI * var2));
  
  return log(beta * exp(-(alpha * alpha) / (2 * var1)) / sqrt(2 * M_PI * var1) + (1 - beta) * exp(-(alpha * alpha) / (2 * var2)) / sqrt(2 * M_PI * var2));
}

//' Function that returns model covariance matrix given hidden and observed parameter values.
//' 
//' @param b21 
inline arma::mat get_covariance_matrix(double b21, double b31, double b32, double c24, double c34, double v1, double v2, double v3, double v4) {
  arma::mat Sigma(3, 3);
  
  // std::cout << "Call to getSigma" << std::endl;
  
  double a321 = b31 + b32 * b21;
  double a324 = c34 + b32 * c24;
  
  Sigma(0, 0) = v1;
  Sigma(0, 1) = Sigma(1, 0) = b21 * v1;
  Sigma(0, 2) = Sigma(2, 0) = a321 * v1;
  Sigma(1, 1) = c24 * c24 * v4 + b21 * b21 * v1 + v2;
  Sigma(1, 2) = Sigma(2, 1) = c24 * a324 * v4 + b21 * a321 * v1 + b32 * v2;
  Sigma(2, 2) = a324 * a324 * v4 + a321 * a321 * v1 + b32 * b32 * v2 + v3;
  
  return Sigma;
}

inline arma::vec get_params(arma::mat Sigma, double sc24, double sc34) {

  arma::mat B(3, 3, arma::fill::zeros);
  arma::vec AV(6, arma::fill::zeros);

  B(0, 0) = 1;
  B(1, 1) = 1 + sc24 * sc24;
  B(1, 2) = B(2, 1) = sc24 * sc34;
  B(2, 2) = 1 + sc34 * sc34;

  arma::mat Q = arma::chol(Sigma, "lower");
  arma::mat U = arma::chol(B, "lower"); 
  arma::mat R = Q * U.i();

  arma::vec sqV = arma::diagvec(R);

  arma::mat I(3, 3, arma::fill::eye);
  
  arma::mat A = I - arma::diagmat(sqV) * U * Q.i();

  AV[0] = A(1, 0);
  AV[1] = A(2, 0);
  AV[2] = A(2, 1);
  AV[3] = sqV[0] * sqV[0];
  AV[4] = sqV[1] * sqV[1];
  AV[5] = sqV[2] * sqV[2];

  return AV;
}

// 
// NumericVector logLikelihoodObservedGradientVarianceScaleFree(
//     NumericMatrix sncsm, int N, 
//     double b12, double b13, double b23, double c24, double c34,
//     double v1, double v2, double v3)

#endif // ROCELL_H
