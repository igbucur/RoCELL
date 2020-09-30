#ifndef ROCELL_H
#define ROCELL_H

#include <armadillo>
#include <assert.h>


//' @author Ioan Gabriel Bucur
arma::vec log_likelihood_gradient_scale_free(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32, 
    double sc24, double sc34, double v1, double v2, double v3);

//' @author Ioan Gabriel Bucur
arma::mat log_likelihood_hessian_scale_free(
    arma::mat sncsm, unsigned N, double sb21, double sb31, double sb32, 
    double sc24, double sc34, double v1, double v2, double v3);

arma::mat log_likelihood_hessian_scale_free_confounders(
    arma::mat sncsm, unsigned N, double b21, double b31, double b32,
    double sc24, double sc34, double v1, double v2, double v3);

inline double log_spike_and_slab_scale_free(double x, double w, double k_slab = 1, double k_spike = 100) {
  
  assert(("Slab precision smaller than spike precision", k_slab <= k_spike));
  
  if (w == 0) {
    return - 0.5 * (k_slab * x * x + log(2 * M_PI) - log(k_slab));
  } else if (w == 1) {
    return - 0.5 * (k_spike * x * x + log(2 * M_PI) - log(k_spike));
  }
  
  // We factor out the slab part to improve numerical stability when both precisions are large
  return -0.5 * (k_slab * x * x + log(2 * M_PI)) + 
    log(w * sqrt(k_slab) + (1 - w) * sqrt(k_spike) * exp(- 0.5 * (k_spike - k_slab) * x * x));
}

inline double log_spike_and_slab(double alpha, double vtail, double varrow, double beta = 0.5, double k1 = 1, double k2 = 1e2) {
  
  double var1 = varrow / vtail / k1;
  double var2 = varrow / vtail / k2;
  
  if (fabs(beta) < 1e-6) return - 0.5 * (alpha * alpha / var2 + log(2 * M_PI * var2));
  
  return log(beta * exp(-(alpha * alpha) / (2 * var1)) / sqrt(2 * M_PI * var1) + (1 - beta) * exp(-(alpha * alpha) / (2 * var2)) / sqrt(2 * M_PI * var2));
}

//' Function that returns 3x3 model covariance matrix given hidden and observed parameter values.
//' 
//' @param b21 
//' @param b31
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

inline arma::vec get_params_scale_free(arma::mat Sigma, double sc24, double sc34) {
  
  // scale-free Omega (I + \tilde{C}\tilde{C}^T)
  arma::mat sOmega(3, 3, arma::fill::zeros);
  // scale-free parameters
  arma::vec spar(6, arma::fill::zeros);
  
  sOmega(0, 0) = 1;
  sOmega(1, 1) = 1 + sc24 * sc24;
  sOmega(1, 2) = sOmega(2, 1) = sc24 * sc34;
  sOmega(2, 2) = 1 + sc34 * sc34;
  
  arma::mat Q = arma::chol(Sigma, "lower");
  arma::mat L = arma::chol(sOmega, "lower"); 
  
  arma::vec sqV = arma::diagvec(Q * L.i());
  
  arma::mat I(3, 3, arma::fill::eye);
  
  // scale-free B matrix (\tilde{B})
  arma::mat sB = I - L * Q.i() * arma::diagmat(sqV);
  
  spar[0] = sB(1, 0);
  spar[1] = sB(2, 0);
  spar[2] = sB(2, 1);
  spar[3] = sqV[0] * sqV[0];
  spar[4] = sqV[1] * sqV[1];
  spar[5] = sqV[2] * sqV[2];
  
  return spar;
}

#endif // ROCELL_H
