## This script constructs the necessary routines for re-creating the Figures
## in the article "Robust Causal Estimation in the Large-Sample Limit without
## Strict Faithfulness" by Ioan Gabriel Bucur, Tom Claassen and Tom Heskes.

# 0 - Setup ---------------------------------------------------------------

library(ggplot2)
library(ggthemes)

source('R/utils.R')
source('R/posterior_functions.R')
Rcpp::sourceCpp("src/RoCELL_Rcpp_wrappers.cpp")

# Create folder where to put article figures
if (!dir.exists("figures")) dir.create("figures")


# 2.2 Spike-and-slab Prior - Figure 1 -------------------------------------

plot_compare_priors <- function() {

  # Function for plotting out the spike and slab densities
  plot_spike_and_slab <- function(sd_spike = sqrt(1e-3), sd_slab = 1, w = 0.5) {
    
    op <- par(pty="s", mar = c(2, 0.1, 0.1, 0.1))
    
    slab <- function(alpha) w * dnorm(alpha, 0, sd_slab)
    spike <- function(alpha) w * dnorm(alpha, 0, sd_spike)
    
    xx <- seq(-3, 3, 1e-4)
    plot(slab(xx) ~ xx, type = 'n', xlab = "", ylab = "", xaxt = 'n', yaxt = 'n', 
         ylim = c(0, spike(0)), asp = 1)
    polygon(slab(xx) ~ xx, col = rgb(0, 0, 1, 0.7) , border = "navyblue") 
    polygon(spike(xx) ~ xx, col = rgb(1, 0, 0, 0.7) , border = "black") 
    axis(1, at = 0, las = 1)
    
    par(op)
  }
  
  pdf('figures/spike_and_slab_prior.pdf')
  plot_spike_and_slab()
  dev.off()
  
  pdf('figures/traditional_prior.pdf')
  plot_spike_and_slab(1e-4, 1e4, 15e-4)
  dev.off()
}

plot_compare_priors()


# 4.1 Illustrative Example - Figure 3 -------------------------------------

reproduce_illustrative_example <- function(
  filename = 'data/illustrative_example_reproduction.RData', 
  num_live_points = 5000, slab_precision = 1, spike_precision = 1000) {
  
  # Parameters for the various scenarios shown in Figure 3
  # (a) ivar = instrumental variable setting without confounding
  # (b) ivar_conf = instrumental variable setting with confounding
  # (c) weak = weak departure from IV setting without confounding
  # (d) weak_conf =  weak departure from IV setting with confounding
  # (e) strong_conf = strong departure from IV setting with confounding
  # (f) adversarial = adversarial scenario with singular coefficients
  parameters <- list(
    'ivar' = c(b21 = 1, b31 = 0, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1),
    'ivar_conf' = c(b21 = 1, b31 = 0, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1),
    'weak' = c(b21 = 1, b31 = 0.05, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1),
    'weak_conf' = c(b21 = 1, b31 = 0.05, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1),
    'strong_conf' = c(b21 = 1, b31 = 1, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1),
    'adversarial' = c(b21 = 1, b31 = 1, b32 = 1, c24 = 1, c34 = 2, v1 = 1, v2 = 1, v3 = 1)
  )
  
  # Collect the scenario names
  scenarios <- names(parameters)
  
  # For each scenario derive the posterior causal effect with RoCELL
  illustrative_example_MultiNest_samples <- lapply(scenarios, function(scenario) {
    
    cat(paste("\n\nScenario = ", scenario))
    
    Sigma <- do.call(get_Sigma, as.list(parameters[[scenario]]))
    
    run_system_command(sprintf(
      "./RoCELL -n %d -slab %g -spike %g -s11 %g -s12 %g -s13 %g -s22 %g -s23 %g -s33 %g",
      num_live_points, slab_precision, spike_precision,
      Sigma[1, 1], Sigma[1, 2], Sigma[1, 3],
      Sigma[2, 2], Sigma[2, 3], Sigma[3, 3]
    ))

    read_MultiNest_samples('output_RoCELL-.txt', Sigma)
  })
  
  # Use the various scenarios to name the elements in the result list
  illustrative_example_MultiNest_samples <- setNames(illustrative_example_MultiNest_samples, scenarios)
  
  # Save the MultiNest results
  save(illustrative_example_MultiNest_samples, file = filename) 
}


plot_illustrative_example <- function(
  data_filename = "data/illustrative_example.RData",
  plot_filename = "figures/illustrative_example_violin_plots.pdf"
) {
  
  load(data_filename)
  
  for (i in seq_along(illustrative_example_MultiNest_samples)) {
    
    item <- data.frame(illustrative_example_MultiNest_samples[[i]][, c('b32', 'w')])
    
    p <- ggplot(item, aes(x = "", y = b32, weight = w)) +
      geom_violin(fill = 'gray', colour = 'darkblue') +
      theme_tufte(base_size = 24) + ylab('') + xlab('') +
      coord_flip(ylim = c(-0.5, 2.5)) +
      ylab(expression(b[32])) +
      xlab("Density") +
      theme(axis.ticks.y = element_blank()) +
      theme(axis.text.y = element_blank()) +
      theme(plot.margin = unit(c(0, 0, 0, 0), "mm"))
    
    ggsave(paste0(dirname(plot_filename), "/", letters[i], "-", basename(plot_filename)), p, width = 5, height = 5)
  }
  
}

plot_illustrative_example()


# 4.3 Weak Departure from IV Setting - Figure 4 ---------------------------

plot_posterior_grid_contour <- function(filename = 'figures/posterior_grid_contour_plot.pdf') {
  
  # Model parameters (Scenario from 4.3, Figure 3c)
  Sigma <- get_Sigma(b21 = 1, b31 = 0.05, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1)
  
  grid_sc24 <- seq(-3.0, 3.0, 0.05)
  grid_sc34 <- seq(-3.0, 3.0, 0.05)
  grid_conf <- expand.grid(grid_sc24, grid_sc34)
  
  grid_values <- apply(grid_conf, 1, function(row) {
    # at the ML solution, Sigma == Sigma_hat
    log_posterior_ML_manifold_scale_free(row[1], row[2], Sigma, samples = 1, k_slab = 1, k_spike = 1e3)
  })
  
  # Derive curve where sb31 == 0 at the ML solution
  s22 <- Sigma[2, 2] - Sigma[1, 2] * Sigma[1, 2] / Sigma[1, 1]
  s23 <- Sigma[2, 3] - Sigma[1, 3] / Sigma[1, 2] * Sigma[2, 2]
  s33 <- Sigma[3, 3] - 2 * Sigma[2, 3] * Sigma[1, 3] / Sigma[1, 2] + Sigma[1, 3] * Sigma[1, 3] * Sigma[2, 2] / (Sigma[1, 2] * Sigma[1, 2])
  f31 <- function(x, y) x * y * sqrt(s22 * s33) - sqrt((1 + x * x) * (1 + y * y)) * s23
  z31 <- mapply(f31, grid_conf[, 1], grid_conf[, 2])
  
  # Derive curve where sb32 == 0 at the ML solution
  t22 <- Sigma[2, 2] - Sigma[1, 2] * Sigma[1, 2] / Sigma[1, 1]
  t23 <- Sigma[2, 3] - Sigma[1, 2] * Sigma[1, 3] / Sigma[1, 1]
  t33 <- Sigma[3, 3] - Sigma[1, 3] * Sigma[1, 3] / Sigma[1, 1]
  f32 <- function(x, y) x * y * sqrt(t22 * t33) - sqrt((1 + x * x) * (1 + y * y)) * t23
  z32 <- mapply(f32, grid_conf[, 1], grid_conf[, 2])
  
  dat <- data.frame(x = grid_conf[, 1], y = grid_conf[, 2], z = grid_values, z31 = z31, z32 = z32)
  
  p <- ggplot(dat, aes(x, y, z = z)) +
    geom_tile(aes(fill = z)) +
    stat_contour(size = 0.3) +
    scale_fill_gradient(name = "Log-Posterior  ", low = "yellow", high = "red") +
    theme_tufte() + coord_fixed() +
    theme(axis.title.x = element_text(size = 20, margin = margin(10, 0, 0, 0)), 
          axis.title.y = element_text(size = 20, margin = margin(0, 5, 0, 0)), 
          axis.text = element_text(size = 10), 
          legend.position = 'top', legend.title = element_text(size = 15)) + 
    xlab(expression(c["2, (2, 3)"])) +
    ylab(expression(c["3, (2, 3)"])) +
    stat_contour(aes(x, y, z = z31), col = 'black', size = 1, breaks = 0) +
    stat_contour(aes(x, y, z = z32), col = 'darkolivegreen', size = 1, breaks = 0)
  
  pdf(file = filename, width = 6.3)
  print(p)
  dev.off()
}

plot_posterior_grid_contour()


# 4.3 Weak Departure from IV Setting - Figure 5 ---------------------------

plot_hessian_term_contour <- function(filename = 'figures/hessian_term_contour_plot.pdf') {
  
  # Model parameters (Scenario from 4.3, Figure 3c)
  Sigma <- get_Sigma(b21 = 1, b31 = 0.05, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1)
  
  grid_sc24 <- seq(-3.0, 3.0, 0.05)
  grid_sc34 <- seq(-3.0, 3.0, 0.05)
  grid_conf <- expand.grid(grid_sc24, grid_sc34)
  
  grid_values <- apply(grid_conf, 1, function(row) {
    log_hessian_term_ML_manifold(row[1], row[2], Sigma, N = 1)
  })
  
  dat <- data.frame(x = grid_conf[, 1], y = grid_conf[, 2], z = grid_values)
  
  p <- ggplot(dat, aes(x, y, z = z)) +
    geom_tile(aes(fill = z)) +
    stat_contour(size = 0.3) +
    scale_fill_gradient(name = "Log-Posterior  ", low = "yellow", high = "red", 
                        breaks = seq(ceiling(min(dat$z)), floor(max(dat$z)), length.out = 3)) +
    theme_tufte() + coord_fixed() +
    theme(axis.title.x = element_text(size = 20, margin = margin(10, 0, 0, 0))) +
    theme(axis.title.y = element_text(size = 20, margin = margin(0, 5, 0, 0))) +
    theme(axis.text = element_text(size = 10), legend.position = 'top', legend.title = element_text(size = 15)) + 
    xlab(expression(c["2, (2, 3)"])) +
    ylab(expression(c["3, (2, 3)"]))
  
  pdf(file = filename, width = 6.3)
  print(p)
  dev.off()
  
}

plot_hessian_term_contour()


# 4.6 Model Selection - Figure 7 ------------------------------------------

reproduce_model_selection <- function(
  filename = "data/model_selection_reproduction.RData", 
  num_live_points = 10000) {

  # Model parameters (Scenario from 4.3, Figure 3d)
  Sigma <- get_Sigma(b21 = 1, b31 = 0.05, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1)
  
  # NOTE: in the original file, the log-posterior was multiplied by the number of samples,
  # which is why the absolute values for the log-evidence differ
  
  model_selection_MultiNest_log_evidences <- as.data.frame(t(sapply(10^(0:12), function(spike_precision) {
    
    # Correct Model: 1 -> 2, 2 -> 3, 4 -> 2, 4 -> 3
    
    run_system_command(sprintf(
      "./RoCELL -o correct -n %d -e 0.3 -spike %g -s11 %g -s12 %g -s13 %g -s22 %g -s23 %g -s33 %g", 
      num_live_points, spike_precision,
      Sigma[1, 1], Sigma[1, 2], Sigma[1, 3],
      Sigma[2, 2], Sigma[2, 3], Sigma[3, 3]
    ))
    lev_correct_model <- read_MultiNest_log_evidence("correct-stats.dat")
    names(lev_correct_model) <- c("Correct", "Correct Error")
    
    # Reverse Model: 1 -> 2, 3 -> 2, 4 -> 2, 4 -> 3
    
    run_system_command(sprintf(
      "./RoCELL -o reverse -n %d -e 0.3 -spike %g -s11 %g -s12 %g -s13 %g -s22 %g -s23 %g -s33 %g", 
      num_live_points, spike_precision,
      Sigma[1, 1], Sigma[1, 3], Sigma[1, 2],
      Sigma[3, 3], Sigma[3, 2], Sigma[2, 2]
    ))
    lev_reverse_model <- read_MultiNest_log_evidence("reverse-stats.dat")
    names(lev_reverse_model) <- c("Reverse", "Reverse Error")
    
    c(Spike = spike_precision, lev_correct_model, lev_reverse_model)
  })))
  
  save(model_selection_MultiNest_log_evidences, file = filename)
}

plot_model_selection <- function(
  plot_filename = "figures/model_selection_plot.pdf",
  data_filename = "data/model_selection.RData") {
  
  load(data_filename)
 
  p <- ggplot(
    model_selection_MultiNest_log_evidences, 
    aes(x = log10(Spike), y = exp(Correct - Reverse))) +
    geom_line(size = 1.5) + 
    scale_x_continuous(breaks = 0:12) +
    ylab('Evidence Ratio') +
    xlab(expression(paste(log[10], " spike precision"))) +
    ggtitle("") + theme_bw(base_size = 16)
  
  ggsave(plot_filename, p)
}

plot_model_selection()


# 4.7 Robustness - Figure 8 -----------------------------------------------

reproduce_spike_variance_robustness <- function(
  filename = 'data/spike_variance_robustness_reproduction.RData', num_live_points = 5000) {
  
  # This covariance corresponds to the weak departure from IV scenario (Figure 3c)
  Sigma <- get_Sigma(b21 = 1, b31 = 0.05, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1)
  
  # We compare results for different spike precisions
  precisions <- paste0(paste0('1e', 0:7))

  # For each different spike precision, we run RoCELL and analyze the posterior
  # of the causal effect, the slab is always N(0, 1)
  spike_variance_robustness_MultiNest_samples <- lapply(precisions, function(spike) {
    
    print(paste("Spike precision = ", spike))
    spike_precision <- as.numeric(spike)
    
    run_system_command(sprintf(
      "./RoCELL -n %d -slab 1 -spike %g -s11 %g -s12 %g -s13 %g -s22 %g -s23 %g -s33 %g",
      num_live_points, spike_precision,
      Sigma[1, 1], Sigma[1, 2], Sigma[1, 3],
      Sigma[2, 2], Sigma[2, 3], Sigma[3, 3]
    ))
    
    read_MultiNest_samples('output_RoCELL-.txt', Sigma)
  })
  
  # Each element in the list is named after the spike precision
  spike_variance_robustness_MultiNest_samples <- setNames(spike_variance_robustness_MultiNest_samples, precisions)
  
  save(spike_variance_robustness_MultiNest_samples, file = filename)
}


plot_spike_variance_robustness <- function(
  data_filename = 'data/spike_variance_robustness.RData',
  plot_filename = "figures/spike_variance_robustness_violin_plot.pdf"
) {
  
  load(data_filename)
  
  powers <- 0:7
  spike_precisions <- paste0('1e', powers)
  
  df <- data.frame()
  
  for (spike in spike_precisions) {
    item <- data.frame(spike_variance_robustness_MultiNest_samples[[spike]][, c('b32', 'w')])
    item$spike <- spike
    df <- rbind(df, item)
  }
  
  # NOTE: WPP bounds were computed with code provided through private correspondence by Dr. Ricardo Silva.
  wpp_bounds <- c(1.0, 1.1)
  
  p <- ggplot(df, aes(x = spike, y = b32, weight = w)) +
    geom_violin(fill = 'gray', colour = 'darkblue') +
    ylab(expression(b[32]))+
    xlab("Spike Variance") +
    coord_flip(ylim = c(-0.5, 1.5)) +
    theme_tufte(base_size = 16) +
    geom_hline(data = data.frame(y = wpp_bounds, lty = rep('WPP bounds', 2)), 
               aes(yintercept = y, linetype = lty)) +
    theme(aspect.ratio = 2) +
    theme(axis.ticks.y = element_blank()) +
    scale_x_discrete(labels = sapply(paste0("10^", -powers), 
                                     function(t) parse(text = t), 
                                     USE.NAMES = FALSE)) +
    scale_linetype_manual(name = "", values = c("WPP bounds" = "dashed")) +
    theme(legend.position = "top") +
    theme(plot.margin = unit(c(0, 0, 0, 0), "mm")) +
    theme(legend.margin = margin(0, 0, 0, 0))
  
  pdf(plot_filename, height = 14)
  print(p)
  dev.off()
  
}

plot_spike_variance_robustness()
