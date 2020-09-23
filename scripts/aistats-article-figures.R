# 0 - Setup ---------------------------------------------------------------


library(Rcpp)
library(ggplot2)
library(ggthemes)

source('R/utils.R')
source('R/posterior_functions.R')
sourceCpp("src/RoCELL_Rcpp_wrappers.cpp")


# 2.2 Spike-and-slab Prior - Figure 1 -------------------------------------


plot_compare_priors <- function() {
  spike_and_slab_plot <- function(sd_spike = sqrt(1e-3), sd_slab = 1, w = 0.5) {
    
    op <- par(pty="s", mar = c(2, 0.1, 0.1, 0.1))
    
    slab <- function(alpha) w * dnorm(alpha, 0, sd_slab)
    spike <- function(alpha) w * dnorm(alpha, 0, sd_spike)
    
    xx <- seq(-3, 3, 1e-4)
    plot(slab(xx) ~ xx, type = 'n', xlab = "", ylab = "", xaxt = 'n', yaxt = 'n', 
         ylim = c(0, spike(0)), asp = 1)
    polygon(slab(xx) ~ xx, col = rgb(0, 0, 1, 0.7) , border="navyblue") 
    polygon(spike(xx) ~ xx, col = rgb(1, 0, 0, 0.7) , border="black") 
    axis(1, at = 0, las = 1)
    
    par(op)
  }
  
  pdf('figures/spike_and_slab_prior.pdf')
  spike_and_slab_plot()
  dev.off()
  
  pdf('figures/traditional_prior.pdf')
  spike_and_slab_plot(1e-4, 1e4, 15e-4)
  dev.off()
}


# 4.3 Weak Departure from IV Setting - Figure 4 ---------------------------

plot_posterior_grid_contour <- function(filename = 'figures/posterior_grid_contour.pdf') {
  
  require(Rcpp) 
  require(ggplot2)
  require(ggthemes)

  samples <- 1
  
  # model parameters
  b21 <- 1
  b31 <- 0.05
  b32 <- 1
  c24 <- 0
  c34 <- 0
  v1 <- v2 <- v3 <- v4 <- 1
  
  true_cov <- getSigma(b21, b31, b32, c24, c34, v1, v2, v3, v4)
  
  
  grid_c24 <- seq(-3.0, 3.0, 0.05)
  grid_c34 <- seq(-3.0, 3.0, 0.05)
  my_grid <- expand.grid(grid_c24, grid_c34)
  
  k <- 1
  spike <- 1e3
  
  system.time(
    grid_posterior_sfc <- apply(my_grid, 1, function(row) {
      # logPosteriorHessianScaleFreeConfounders(row[1], row[2], true_cov, true_cov, samples)
      logPosteriorManifoldScaleFreeConfounders(row[1], row[2], true_cov, true_cov, samples, k, k * spike)
    }))
  
  s22 <- true_cov[2, 2] - true_cov[1, 2] * true_cov[1, 2] / true_cov[1, 1]
  s23 <- true_cov[2, 3] - true_cov[1, 3] / true_cov[1, 2] * true_cov[2, 2]
  s33 <- true_cov[3, 3] - 2 * true_cov[2, 3] * true_cov[1, 3] / true_cov[1, 2] + true_cov[1, 3] * true_cov[1, 3] * true_cov[2, 2] / (true_cov[1, 2] * true_cov[1, 2])
  f13 <- function(x, y) x * y * sqrt(s22 * s33) - sqrt((1 + x * x) * (1 + y * y)) * s23
  z13 <- mapply(f13, my_grid[, 1], my_grid[, 2])
  
  t22 <- true_cov[2, 2] - true_cov[1, 2] * true_cov[1, 2] / true_cov[1, 1]
  t23 <- true_cov[2, 3] - true_cov[1, 2] * true_cov[1, 3] / true_cov[1, 1]
  t33 <- true_cov[3, 3] - true_cov[1, 3] * true_cov[1, 3] / true_cov[1, 1]
  f23 <- function(x, y) x * y * sqrt(t22 * t33) - sqrt((1 + x * x) * (1 + y * y)) * t23
  z23 <- mapply(f23, my_grid[, 1], my_grid[, 2])
  
  dat <- data.frame(x = my_grid[,1], y = my_grid[,2], z = grid_posterior_sfc, z13 = z13, z23 = z23)
  
  
  p <- ggplot(dat, aes(x, y, z = z)) +
    geom_tile(aes(fill = z)) +
    stat_contour(size = 0.3) +
    scale_fill_gradient(name = "Log-Posterior  ", low = "yellow", high = "red") + #, breaks = seq(ceiling(min(dat$z)), floor(max(dat$z)), length.out = 3)) +
    theme_tufte() + coord_fixed() +
    theme(axis.title.x = element_text(size = 20, margin = margin(10, 0, 0, 0)), axis.title.y = element_text(size = 20, margin = margin(0, 5, 0, 0)), axis.text = element_text(size = 10), legend.position = 'top', legend.title = element_text(size = 15)) + 
    xlab(expression(c["2, (2, 3)"])) +
    ylab(expression(c["3, (2, 3)"])) +
    stat_contour(aes(x, y, z = z13), col = 'black', size = 1, breaks = 0) +
    stat_contour(aes(x, y, z = z23), col = 'darkolivegreen', size = 1, breaks = 0)
  
  pdf(file = filename, width = 6.3)
  print(p)
  dev.off()
}




# 4.3 Weak Departure from IV Setting - Figure 5 ---------------------------

plot_hessian_term <- function(filename = 'figures/hessian_term_plot.pdf') {
  
  
  samples <- 1
  
  # model parameters
  b21 <- 1
  b31 <- 0.05
  b32 <- 1
  c24 <- 0
  c34 <- 0
  v1 <- v2 <- v3 <- v4 <- 1
  
  true_cov <- getSigma(b21, b31, b32, c24, c34, v1, v2, v3, v4)
  
  grid_c24 <- seq(-3.0, 3.0, 0.05)
  grid_c34 <- seq(-3.0, 3.0, 0.05)
  my_grid <- expand.grid(grid_c24, grid_c34)
  
  system.time(
    grid_posterior_sfc <- apply(my_grid, 1, function(row) {
      logPosteriorHessianScaleFreeConfounders(row[1], row[2], true_cov, true_cov, samples)
    }))
  
  
  dat <- data.frame(x = my_grid[, 1], y = my_grid[, 2], z = grid_posterior_sfc)
  
  p <- ggplot(dat, aes(x, y, z = z)) +
    geom_tile(aes(fill = z)) +
    stat_contour(size = 0.3) +
    scale_fill_gradient(name = "Log-Posterior  ", low = "yellow", high = "red", breaks = seq(ceiling(min(dat$z)), floor(max(dat$z)), length.out = 3)) +
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

# 4.1 Illustrative Example - Figure 3 -------------------------------------

reproduce_scenario_results <- function(filename= 'results/scenario_results.rds') {
  
  parameters <- list(
    'ivar' = c(b21 = 1, b31 = 0, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1, v4 = 1),
    'ivar_conf' = c(b21 = 1, b31 = 0, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1, v4 = 1),
    'weak' = c(b21 = 1, b31 = 0.05, b32 = 1, c24 = 0, c34 = 0, v1 = 1, v2 = 1, v3 = 1, v4 = 1),
    'weak_conf' = c(b21 = 1, b31 = 0.05, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1, v4 = 1),
    'strong_conf' = c(b21 = 1, b31 = 1, b32 = 1, c24 = 1, c34 = 1, v1 = 1, v2 = 1, v3 = 1, v4 = 1),
    'adversarial' = c(b21 = 1, b31 = 1, b32 = 1, c24 = 1, c34 = 2, v1 = 1, v2 = 1, v3 = 1, v4 = 1)
  )
  
  scenarios <- names(parameters)
  
  MN_results <- lapply(scenarios, function(scenario) {
    
    cat(paste("\n\nScenario = ", scenario))
    
    covariance <- do.call(getSigma, as.list(parameters[[scenario]]))
    
    run_system_command(sprintf("./RoCELL.exe 1000 2 1 1000 %g %g %g %g %g %g",
                   covariance[1, 1], covariance[1, 2], covariance[1, 3],
                   covariance[2, 2], covariance[2, 3], covariance[3, 3]))
    
    MN_results <- read_MultiNest_output_Gaussian_prior('output_RoCELL-.txt', covariance)
    
  })
  
  saveRDS(file = filename, MN_results)  
}


plot_posterior_violin_scenario <- function(
  data_filename = "results/scenario_results.rds",
  plot_filename = "figures/scenario_violin_plot_2.pdf"
) {
  
  require(ggplot2)
  require(ggthemes)
  require(grid)
  require(gridExtra)
  
  MN_results <- readRDS(data_filename)
  # load(data_filename)
  
  plots <- list()
  
  for (i in seq_along(MN_results)) {
    
    item <- data.frame(MN_results[[i]][, c('b32', 'w')])
    
    plots[[i]] <- ggplot(item, aes(x = "", y = b32, weight = w)) +
      geom_violin(fill = 'gray', colour = 'darkblue') +
      theme_tufte(base_size = 24) + ylab('') + xlab('') +
      #ggtitle(paste0(letters[i], ')')) +
      coord_flip(ylim = c(-0.5, 2.5)) +
      ylab(expression(b[32])) +
      xlab("Density") +
      #theme(aspect.ratio = 1) +
      theme(axis.ticks.y = element_blank()) +
      theme(axis.text.y = element_blank()) +
      theme(plot.margin = unit(c(0, 0, 0, 0), "mm"))
    
    ggsave(paste0(dirname(plot_filename), "/", letters[i], "-", basename(plot_filename)), plots[[i]], width = 5, height = 5)
  }
  
}





# 4.6 Model Selection - Figure 7 ------------------------------------------

model_selection_plot <- function(filename = "figures/model_selection.pdf") {
  
  require(ggplot2)
  require(ggthemes)
  require(reshape2)
  
  dirrev <- read.csv('data/direct_vs_reverse.csv')
  melted_dirrev <- melt(dirrev[, c('Spike.Posterior', 'Correct', 'Reverse')], id = 'Spike.Posterior')
  
  model_selection_plot <- ggplot(dirrev, aes(x = log10(Spike.Posterior), y = exp(Correct - Reverse))) +
    geom_line(size = 1.5) + 
    scale_x_continuous(breaks = 0:12) +
    ylab('Evidence Ratio') +
    xlab(expression(paste(log[10], " spike precision"))) +
    ggtitle("") + theme_bw(base_size = 16)
  
  ggsave(filename, model_selection_plot)
}


# 4.7 Robustness - Figure 8 -----------------------------------------------

reproduce_robustness_spike_width <- function(filename = 'results/robustness_spike_width.RData') {
  # require(coda)
  # source('R/covariance_decomposition_functions.R')
  
  covariance <- getSigma(1, 0.05, 1, 0, 0, 1, 1, 1, 1)
  correlation <- cov2cor(covariance)
  
  
  widths <- paste0(paste0('1e', 0:7))
  
  
  
  MN_results <- lapply(widths, function(spike) {
    
    print(paste("Spike=", spike))
    
    spike_num <- as.numeric(spike)
    
    run_system_command(sprintf("./RoCELL.exe 1000 2 1 %g %g %g %g %g %g %g", spike_num,
                   covariance[1, 1], covariance[1, 2], covariance[1, 3],
                   covariance[2, 2], covariance[2, 3], covariance[3, 3]))
    
    read_MultiNest_output_Gaussian_prior('output_RoCELL-.txt', covariance)
  })
  
  MN_results <- setNames(MN_results, widths)
  
  save(file = filename, MN_results, widths, covariance, correlation)
}


plot_posterior_violin_spike_width <- function(
  data_filename = 'results/robustness_spike_width.RData',
  plot_filename = "figures/robustness_violin_plot.pdf"
) {
  
  require(ggplot2)
  require(ggthemes)
  
  load(data_filename)
  
  powers <- 0:7
  spike_precisions <- paste0('1e', powers)
  
  df <- data.frame()
  
  for (spike in spike_precisions) {
    item <- data.frame(MN_results[[spike]][, c('b32', 'w')])
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
