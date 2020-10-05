#' A dataset containing simulated samples for reproducing Figure 3
#' Section 4 of the 2017 AISTATS paper.
#'
#' @format A list containing six coda::mcmc objects, which represent the
#' posterior samples for each scenario.
"illustrative_example_MultiNest_samples"

#' A dataset containing simulated samples for reproducing Figure 7
#' Section 4 of the 2017 AISTATS paper.
#'
#' @format A data frame containing the estimated log-evidences and standard errors.
#' \describe{
#'   \item{Spike}{spike component precision}
#'   \item{Correct}{Log-evidence in the correct direction (of the generating model)}
#'   \item{Correct Error}{Standard error for estimated log-evidence}
#'   \item{Reverse}{Log-evidence in the reverse direction (of the generating model)}
#'   \item{Reverse Error}{Standard error for estimated log-evidence}
#' }
"model_selection_MultiNest_log_evidences"

#' A dataset containing simulated samples for reproducing Figure 8 in
#' Section 4 of the 2017 AISTATS paper.
#'
#' @format A list containing ten coda::mcmc objects, which represent the
#' posterior samples for each spike width (variance).
"spike_variance_robustness_MultiNest_samples"