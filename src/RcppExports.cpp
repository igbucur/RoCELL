// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// Rcpp_log_spike_and_slab_scale_free
double Rcpp_log_spike_and_slab_scale_free(double x, double w, double k_slab, double k_spike);
RcppExport SEXP _RoCELL_Rcpp_log_spike_and_slab_scale_free(SEXP xSEXP, SEXP wSEXP, SEXP k_slabSEXP, SEXP k_spikeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type k_slab(k_slabSEXP);
    Rcpp::traits::input_parameter< double >::type k_spike(k_spikeSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_log_spike_and_slab_scale_free(x, w, k_slab, k_spike));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_get_params_scale_free
arma::vec Rcpp_get_params_scale_free(arma::mat Sigma_hat, double sc24, double sc34);
RcppExport SEXP _RoCELL_Rcpp_get_params_scale_free(SEXP Sigma_hatSEXP, SEXP sc24SEXP, SEXP sc34SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Sigma_hat(Sigma_hatSEXP);
    Rcpp::traits::input_parameter< double >::type sc24(sc24SEXP);
    Rcpp::traits::input_parameter< double >::type sc34(sc34SEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_get_params_scale_free(Sigma_hat, sc24, sc34));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_log_posterior_RoCELL
double Rcpp_log_posterior_RoCELL(arma::mat Sigma_hat, unsigned N, double sb21, double sb31, double sb32, double sc24, double sc34, double v1, double v2, double v3, double slab_precision, double spike_precision, double w32);
RcppExport SEXP _RoCELL_Rcpp_log_posterior_RoCELL(SEXP Sigma_hatSEXP, SEXP NSEXP, SEXP sb21SEXP, SEXP sb31SEXP, SEXP sb32SEXP, SEXP sc24SEXP, SEXP sc34SEXP, SEXP v1SEXP, SEXP v2SEXP, SEXP v3SEXP, SEXP slab_precisionSEXP, SEXP spike_precisionSEXP, SEXP w32SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Sigma_hat(Sigma_hatSEXP);
    Rcpp::traits::input_parameter< unsigned >::type N(NSEXP);
    Rcpp::traits::input_parameter< double >::type sb21(sb21SEXP);
    Rcpp::traits::input_parameter< double >::type sb31(sb31SEXP);
    Rcpp::traits::input_parameter< double >::type sb32(sb32SEXP);
    Rcpp::traits::input_parameter< double >::type sc24(sc24SEXP);
    Rcpp::traits::input_parameter< double >::type sc34(sc34SEXP);
    Rcpp::traits::input_parameter< double >::type v1(v1SEXP);
    Rcpp::traits::input_parameter< double >::type v2(v2SEXP);
    Rcpp::traits::input_parameter< double >::type v3(v3SEXP);
    Rcpp::traits::input_parameter< double >::type slab_precision(slab_precisionSEXP);
    Rcpp::traits::input_parameter< double >::type spike_precision(spike_precisionSEXP);
    Rcpp::traits::input_parameter< double >::type w32(w32SEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_log_posterior_RoCELL(Sigma_hat, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3, slab_precision, spike_precision, w32));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_log_likelihood_gradient_scale_free
arma::vec Rcpp_log_likelihood_gradient_scale_free(arma::mat Sigma_hat, unsigned N, double sb21, double sb31, double sb32, double sc24, double sc34, double v1, double v2, double v3);
RcppExport SEXP _RoCELL_Rcpp_log_likelihood_gradient_scale_free(SEXP Sigma_hatSEXP, SEXP NSEXP, SEXP sb21SEXP, SEXP sb31SEXP, SEXP sb32SEXP, SEXP sc24SEXP, SEXP sc34SEXP, SEXP v1SEXP, SEXP v2SEXP, SEXP v3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Sigma_hat(Sigma_hatSEXP);
    Rcpp::traits::input_parameter< unsigned >::type N(NSEXP);
    Rcpp::traits::input_parameter< double >::type sb21(sb21SEXP);
    Rcpp::traits::input_parameter< double >::type sb31(sb31SEXP);
    Rcpp::traits::input_parameter< double >::type sb32(sb32SEXP);
    Rcpp::traits::input_parameter< double >::type sc24(sc24SEXP);
    Rcpp::traits::input_parameter< double >::type sc34(sc34SEXP);
    Rcpp::traits::input_parameter< double >::type v1(v1SEXP);
    Rcpp::traits::input_parameter< double >::type v2(v2SEXP);
    Rcpp::traits::input_parameter< double >::type v3(v3SEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_log_likelihood_gradient_scale_free(Sigma_hat, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_log_likelihood_hessian_scale_free
arma::mat Rcpp_log_likelihood_hessian_scale_free(arma::mat Sigma_hat, unsigned N, double sb21, double sb31, double sb32, double sc24, double sc34, double v1, double v2, double v3);
RcppExport SEXP _RoCELL_Rcpp_log_likelihood_hessian_scale_free(SEXP Sigma_hatSEXP, SEXP NSEXP, SEXP sb21SEXP, SEXP sb31SEXP, SEXP sb32SEXP, SEXP sc24SEXP, SEXP sc34SEXP, SEXP v1SEXP, SEXP v2SEXP, SEXP v3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Sigma_hat(Sigma_hatSEXP);
    Rcpp::traits::input_parameter< unsigned >::type N(NSEXP);
    Rcpp::traits::input_parameter< double >::type sb21(sb21SEXP);
    Rcpp::traits::input_parameter< double >::type sb31(sb31SEXP);
    Rcpp::traits::input_parameter< double >::type sb32(sb32SEXP);
    Rcpp::traits::input_parameter< double >::type sc24(sc24SEXP);
    Rcpp::traits::input_parameter< double >::type sc34(sc34SEXP);
    Rcpp::traits::input_parameter< double >::type v1(v1SEXP);
    Rcpp::traits::input_parameter< double >::type v2(v2SEXP);
    Rcpp::traits::input_parameter< double >::type v3(v3SEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_log_likelihood_hessian_scale_free(Sigma_hat, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_log_likelihood_hessian_scale_free_confounders
arma::mat Rcpp_log_likelihood_hessian_scale_free_confounders(arma::mat Sigma_hat, unsigned N, double b21, double b31, double b32, double sc24, double sc34, double v1, double v2, double v3);
RcppExport SEXP _RoCELL_Rcpp_log_likelihood_hessian_scale_free_confounders(SEXP Sigma_hatSEXP, SEXP NSEXP, SEXP b21SEXP, SEXP b31SEXP, SEXP b32SEXP, SEXP sc24SEXP, SEXP sc34SEXP, SEXP v1SEXP, SEXP v2SEXP, SEXP v3SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Sigma_hat(Sigma_hatSEXP);
    Rcpp::traits::input_parameter< unsigned >::type N(NSEXP);
    Rcpp::traits::input_parameter< double >::type b21(b21SEXP);
    Rcpp::traits::input_parameter< double >::type b31(b31SEXP);
    Rcpp::traits::input_parameter< double >::type b32(b32SEXP);
    Rcpp::traits::input_parameter< double >::type sc24(sc24SEXP);
    Rcpp::traits::input_parameter< double >::type sc34(sc34SEXP);
    Rcpp::traits::input_parameter< double >::type v1(v1SEXP);
    Rcpp::traits::input_parameter< double >::type v2(v2SEXP);
    Rcpp::traits::input_parameter< double >::type v3(v3SEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_log_likelihood_hessian_scale_free_confounders(Sigma_hat, N, b21, b31, b32, sc24, sc34, v1, v2, v3));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RoCELL_Rcpp_log_spike_and_slab_scale_free", (DL_FUNC) &_RoCELL_Rcpp_log_spike_and_slab_scale_free, 4},
    {"_RoCELL_Rcpp_get_params_scale_free", (DL_FUNC) &_RoCELL_Rcpp_get_params_scale_free, 3},
    {"_RoCELL_Rcpp_log_posterior_RoCELL", (DL_FUNC) &_RoCELL_Rcpp_log_posterior_RoCELL, 13},
    {"_RoCELL_Rcpp_log_likelihood_gradient_scale_free", (DL_FUNC) &_RoCELL_Rcpp_log_likelihood_gradient_scale_free, 10},
    {"_RoCELL_Rcpp_log_likelihood_hessian_scale_free", (DL_FUNC) &_RoCELL_Rcpp_log_likelihood_hessian_scale_free, 10},
    {"_RoCELL_Rcpp_log_likelihood_hessian_scale_free_confounders", (DL_FUNC) &_RoCELL_Rcpp_log_likelihood_hessian_scale_free_confounders, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_RoCELL(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
