% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{Rcpp_log_likelihood_hessian_scale_free_confounders}
\alias{Rcpp_log_likelihood_hessian_scale_free_confounders}
\title{Compute log-likelihood hessian wrt scaled confounding coefficients and
structural coefficients in original scale (unscaled).}
\usage{
Rcpp_log_likelihood_hessian_scale_free_confounders(
  Sigma_hat,
  N,
  b21,
  b31,
  b32,
  sc24,
  sc34,
  v1,
  v2,
  v3
)
}
\arguments{
\item{Sigma_hat}{sample covariance matrix}

\item{N}{number of samples}

\item{b21}{structural coefficient for 1 -> 2}

\item{b31}{structural coefficient for 1 -> 3}

\item{b32}{structural coefficient for 1 -> 3}

\item{sc24}{scaled confounding coefficient for 4 -> 2}

\item{sc34}{scaled confounding coefficient for 4 -> 3}

\item{v1}{intrinsic variance of 1}

\item{v2}{intrinsic variance of 2}

\item{v3}{intrinsic variance of 3}
}
\value{
log-likelihood Hessian matrix (8x8) with the parameters ordered as above
}
\description{
Compute log-likelihood hessian wrt scaled confounding coefficients and
structural coefficients in original scale (unscaled).
}
