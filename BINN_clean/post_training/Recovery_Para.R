# Choose a cross-validation case with middle SOC recovery correlation
# Plot the scatter plot of BINN predicted parameter vs the prescribed parameter (for each recovered parameter)
# Plot a box plot showing the mean correlation of the recovered parameters and the mean correlation of the SOC across the 10 cross-validation cases

## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
# library(jcolors)
library(gridExtra)
library(viridis)
library(sf)
library(sp)
library(GGally)
library(raster)
library(proj4)
library(scales)
library(ncdf4)
library(jpeg)
library(tidyverse)
library(magick)


setwd('D:/Research/BINN/BINN_output/plot/')

## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "white", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))

#############################################################################
# function to increase vertical spacing between legend keys
#############################################################################
# @clauswilke
draw_key_polygon3 <- function(data, params, size) {
  lwd <- min(data$size, min(size) / 4)
  
  grid::rectGrob(
    width = grid::unit(0.6, "npc"),
    height = grid::unit(0.6, "npc"),
    gp = grid::gpar(
      col = data$colour,
      fill = alpha(data$fill, data$alpha),
      lty = data$linetype,
      lwd = lwd * .pt,
      linejoin = "mitre"
    ))
}

# register new key drawing function, 
# the effect is global & persistent throughout the R session
GeomBar$draw_key = draw_key_polygon3

#############################################################################
# Data Path
#############################################################################
cross_validation_folder = 'Recovery_Cross_Validation_10_Fold'
cross_validation_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/')
cross_validation_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/Output/')
if (!dir.exists(cross_validation_dir_output)) {
  dir.create(cross_validation_dir_output)
}

# Define the cross validation types
cross_validation_experiments = c('Random_Seed', 'Cross_Validation')
cross_validation_experiments = cross_validation_experiments[2]
# Get the list of all the folders in the cross validation directory
recovery_experiments = list.dirs(cross_validation_dir_input, full.names = FALSE, recursive = FALSE)
# Exclude the folder of Output
recovery_experiments = recovery_experiments[!recovery_experiments %in% 'Output']

# PRODA data path
# data_dir_PRODA = 'D:/Nutstore/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
# data_dir_loc = 'D:/Nutstore/hx293/Research_Data/BINN/Server_Script/post_training/component_calculation/'
# proda_para_input = 'D:/Research/Binn/PRODA_Results/'
data_dir_PRODA = 'D:/Research/BINN/PRODA_Results/soc_component_proda/soc_component_proda/'
data_dir_loc = 'D:/Research/BINN/PRODA_Results/component_calculation/'
proda_para_input = 'D:/Research/BINN/PRODA_Results/'


# create output folder if not exist
if (!dir.exists(cross_validation_dir_output)) {
  dir.create(cross_validation_dir_output)
}

# Define the index of the parameters that is trained
para_names = c('diffus', 'cryo', 'q10', 'efolding', 
               'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 
               'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 
               'w-scaling', 'beta')

## Loop through each recovery experiment
for (recovery_idx in 1:length(recovery_experiments)) {

if (recovery_experiments[recovery_idx] == 'All') {
  para_index = 1:21
  plot_col = 4
  plot_col_height = 80
  plot_col_width = 60
} else if (recovery_experiments[recovery_idx] == 'Sixteen') {
  para_index = c(0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20)
  para_index = para_index + 1
} else if (recovery_experiments[recovery_idx] == 'Four') {
  para_index = c(3, 9, 14, 19)
  para_index = para_index + 1
  plot_col = 3
  plot_col_height = 30
  plot_col_width = 60
} else if (recovery_experiments[recovery_idx] == 'Five') {
  para_index = c(3, 9, 14, 19, 20)
  para_index = para_index + 1
  plot_col = 3
  plot_col_height = 35
  plot_col_width = 70
} 

# Get the cross validation directories for the current recovery experiment
cross_validation_dir_list = list.dirs(paste0(cross_validation_dir_input, recovery_experiments[recovery_idx]), full.names = FALSE, recursive = FALSE)

if (cross_validation_experiments == 'Random_Seed') {
  # Get the list of random seeds (the last part of the folder name after =)
  random_seed_list = sapply(strsplit(cross_validation_dir_list, split = '='), function(x) x[length(x)])
  random_seed_list = as.numeric(random_seed_list)
  # Error if the random seed list is not unique
  if (length(unique(random_seed_list)) != length(random_seed_list)) {
    stop('Random seed list is not unique!')
  }
  } else if (cross_validation_experiments == 'Cross_Validation') {
    # Get the list of random seeds (the last part of the folder name after _)
    random_seed_list = sapply(strsplit(cross_validation_dir_list, split = '_'), function(x) x[length(x)-1])
    random_seed_list = sapply(strsplit(random_seed_list, split = '='), function(x) x[length(x)])
    random_seed_list = as.numeric(random_seed_list)
    # Error if the random seed list is not unique
    if (length(unique(random_seed_list)) != length(random_seed_list)) {
      stop('Cross validation list is not unique!')
    }
  } else {
    stop('Unsupported cross validation experiment!')
  }

#############################################################################
# Cross Validation SOC NSE and Parameter Correlation
#############################################################################
cross_validation_soc_nse = array(NA, dim = c(length(cross_validation_dir_list), 1))
cross_validation_soc_nse = cbind(cross_validation_dir_list, cross_validation_soc_nse)
cross_validation_para_correlation = array(NA, dim = c(length(cross_validation_dir_list), length(para_names)))
cross_validation_para_mean_correlation = array(NA, dim = c(length(cross_validation_dir_list), 1))

## PRODA ##
# Initialize an empty data frame for storing the combined results
PRODA_para <- data.frame()
# Read and combine the predicted parameters from the files
for (i in 1:9) {
  # Read the profile id file
  nn_site_loc_temp <- read_csv(paste0(proda_para_input, '/nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_', i, '.csv'), col_names = FALSE)

  # Read the predicted parameters file
  nn_site_para_temp <- read_csv(paste0(proda_para_input, '/nn_para_result_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_', i, '.csv'), col_names = FALSE)
  colnames(nn_site_para_temp) <- para_names
  
  # Combine profile id with its parameters
  if (i == 1) {
    PRODA_para <- nn_site_loc_temp
    colnames(PRODA_para)[1] <- 'profile_id'
    PRODA_para <- bind_cols(PRODA_para, nn_site_para_temp)
  } else {
    PRODA_para <- bind_cols(PRODA_para, nn_site_para_temp)
  }
}

# head(PRODA_para)

# Calculate mean for each parameter across profiles
for (i in 1:21) {
  PRODA_para <- PRODA_para %>%
    mutate(!!paste0('mean_', i) := rowMeans(select(., starts_with(para_names[i]))))
}

# Drop the original parameter columns
keep_cols <- grep('mean', colnames(PRODA_para))
# Also keep the profile id column
keep_cols <- c(1, keep_cols)
# Drop the original parameter columns and keep the mean columns
PRODA_para <- PRODA_para[ , keep_cols]

# Print the head of the dataframe
head(PRODA_para)
# Check the shape of the dataframe
dim(PRODA_para)


## BINN ##
# Initialize a dataframe to store the parameters for plotting
para_fold_plot = data.frame(value = numeric(0), 
                            parameter = character(0), 
                            random_seed = numeric(0))
for (i in 1:length(cross_validation_dir_list)) {
    # Test SOC
    binn_simu_soc = read.csv(paste(cross_validation_dir_input, recovery_experiments[recovery_idx], '/', cross_validation_dir_list[i], '/Test/nn_test_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc = data.matrix(binn_simu_soc)
    binn_obs_soc = read.csv(paste(cross_validation_dir_input, recovery_experiments[recovery_idx], '/', cross_validation_dir_list[i], '/nn_obs_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_obs_soc = data.matrix(binn_obs_soc)
    valid_soc_loc = which(is.na(binn_simu_soc[ , 1]) == 0 & is.na(binn_obs_soc[ , 1]) == 0)
    # Calculate the SOC NSE
    middle_soc_corr = cbind(as.vector(binn_simu_soc[valid_soc_loc, ]), as.vector(binn_obs_soc[valid_soc_loc, ]))/1000
    middle_soc_corr = data.frame(middle_soc_corr[which(is.na(middle_soc_corr[ , 1]) == 0), ])
    colnames(middle_soc_corr) = c('binn', 'obs')
    cross_validation_soc_nse[i, 2] = 1 - sum((middle_soc_corr$binn - middle_soc_corr$obs)^2)/sum((mean(middle_soc_corr$obs) - middle_soc_corr$obs)^2)


    # Test parameters
    binn_para = read.csv(paste(cross_validation_dir_input, recovery_experiments[recovery_idx], '/', cross_validation_dir_list[i], '/Test/nn_test_best_pred_para_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_para = data.matrix(binn_para)
    valid_para_loc_temp = which(is.na(binn_para[ , 1]) == 0)
    valid_para_loc <- intersect(valid_para_loc_temp, PRODA_para$profile_id)
    # Calculate the parameter correlation
    for (ipara in 1:length(para_names)) {
        if (!(ipara %in% para_index)) {
            next
        } else {
            current_data = cbind(PRODA_para[PRODA_para$profile_id %in% valid_para_loc, ipara+1], binn_para[valid_para_loc, ipara])
            current_data = data.frame(current_data)
            # Build a data frame to store the current parameter data for future plotting
            para_fold_plot = rbind(para_fold_plot, data.frame(value = current_data[ , 2], 
                                                               parameter = para_names[ipara], 
                                                                random_seed = as.numeric(random_seed_list[i])))
            colnames(current_data) = c('proda', 'binn')           
            cross_validation_para_correlation[i, ipara] = cor(current_data$proda, current_data$binn, use = 'complete.obs')
        }
    }
}

cross_validation_para_mean_correlation = apply(cross_validation_para_correlation, 1, mean, na.rm = TRUE)

# Select the cross-validation case with the middle SOC recovery correlation
# Sort the cross-validation cases by the SOC NSE
selected_cross_validation_case = cross_validation_soc_nse[order(cross_validation_soc_nse[ , 2], decreasing = TRUE), 1][5]

##############################################
# Recovered Parameters vs. Observed Parameters
##############################################
# Scatter plot function for unifying the legend
make_scatter_plot <- function(middle_data_corr, para_name, corr_value, limit_proda, limit_binn, is_soc = FALSE) {
  p <- ggplot() + 
    stat_bin_hex(data = middle_data_corr, aes(x = proda, y = binn), bins = if(is_soc) 100 else 30) +
    geom_abline(slope = 1, intercept = 0, size = if(is_soc) 1 else 2, color = 'black') +
    theme_classic()
    
  if (is_soc) {
    p <- p +
      scale_y_continuous(limits = c(0.1, 1000), trans = 'log10', 
                        labels = trans_format('log10', math_format(10^.x))) + 
      scale_x_continuous(limits = c(0.1, 1000), trans = 'log10', 
                        labels = trans_format('log10', math_format(10^.x))) +
      labs(title = paste('BINN vs SOC (NSE= ', round(corr_value, 2), ')', sep = ''),
           x = expression(paste('Synthetic SOC (kg C m'^'-3', ')', sep = '')), 
           y = expression(paste('BINN simulation (kg C m'^'-3', ')', sep = '')))
  } else {
    p <- p +
      scale_x_continuous(trans = 'identity', limits = limit_proda) +
      scale_y_continuous(trans = 'identity', limits = limit_binn) +
      labs(title = paste(para_name, ' (r= ', round(corr_value, 2), ')', sep = ''),
           x = 'Prescribed', y = 'BINN')
  }
  
  p <- p +
    scale_fill_gradientn(name = 'Count', colors = viridis(7), 
                        limits = c(if(is_soc) 1 else 0, 20), 
                        trans = 'identity', oob = scales::squish) +
    theme(plot.title = element_text(hjust = 0.5, size = 80),
          plot.margin = unit(c(0.2, 0.5, 0.8, 0.2), 'inch'),
          axis.text = element_text(size = 60, color = 'black'),
          axis.title = element_text(size = 60),
          axis.line = element_line(size = 1),
          axis.ticks = element_line(size = 1, color = 'black'),
          axis.ticks.length = unit(0.12, 'inch'),
          legend.position = "none")
  
  return(p)
}

# Create a standalone legend with larger dimensions and font
legend_plot <- ggplot() + 
  stat_bin_hex(data = data.frame(x = 1, y = 1), aes(x = x, y = y), bins = 30) +
  scale_fill_gradientn(name = ' ', colors = viridis(7), 
                      limits = c(0, 20), trans = 'identity', 
                      oob = scales::squish) +
  theme_void() +
  guides(fill = guide_colorbar(
    direction = 'vertical',
    barwidth = 8,           
    barheight = 50,       
    # title.position = 'top',
    # title.hjust = 0.5,
    # title.vjust = 1,
    label.hjust = 0.5,
    frame.linewidth = 1
  )) +
  theme(legend.position = c(0.5, -0), 
        legend.justification = c(0.5, 0.5),
        legend.text = element_text(size = 60),      
        legend.title = element_text(size = 70),      
        legend.key.size = unit(3, "cm"))            

# # Extract the legend
# shared_legend <- get_legend(legend_plot)


# First a few plots: scatter plot of the predicted parameter vs the prescribed parameter for the selected cross-validation case
binn_para <- read.csv(paste(cross_validation_dir_input,  recovery_experiments[recovery_idx], '/', selected_cross_validation_case, '/Test/nn_test_best_pred_para_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_para = data.matrix(binn_para)

valid_profile_loc_temp <- which(!is.na(binn_para[, 1]))
valid_profile_loc <- intersect(valid_profile_loc_temp, PRODA_para$profile_id)

p_para_performance_list = list()
plot_idx = 1

ipara = 4
for (ipara in 1:length(para_names)) {
    if (!(ipara %in% para_index)) {
        next
    } else {
        middle_data_corr = cbind(PRODA_para[PRODA_para$profile_id %in% valid_profile_loc, ipara+1], binn_para[valid_profile_loc, ipara])
        middle_data_corr = data.frame(middle_data_corr)
        colnames(middle_data_corr) = c('proda', 'binn')

        limit_proda = quantile(PRODA_para[PRODA_para$profile_id %in% valid_profile_loc, ipara+1], probs = c(0, 1), na.rm = TRUE)
        limit_binn = quantile(binn_para[valid_profile_loc, ipara], probs = c(0, 1), na.rm = TRUE)

        limit_common = c(min(limit_proda[1], limit_binn[1]), max(limit_proda[2], limit_binn[2]))

        # Calculate the correlation
        corr_process_middle = cor.test(middle_data_corr$proda, middle_data_corr$binn, na.rm = TRUE)

        p_para_performance = make_scatter_plot(
            middle_data_corr, para_names[ipara], corr_process_middle$estimate, 
            limit_common, limit_common)

        p_para_performance_list[[plot_idx]] = p_para_performance
        plot_idx = plot_idx + 1
    }
}

# Second plot: scatter plot of BINN predicted SOC vs actual SOC for the selected cross-validation case
binn_simu_soc = read.csv(paste(cross_validation_dir_input, recovery_experiments[recovery_idx], '/', selected_cross_validation_case, '/Test/nn_test_best_simu_soc_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_simu_soc = data.matrix(binn_simu_soc)
binn_obs_soc = read.csv(paste(cross_validation_dir_input, recovery_experiments[recovery_idx], '/', selected_cross_validation_case, '/nn_obs_soc_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_soc = data.matrix(binn_obs_soc)
valid_soc_loc = which(is.na(binn_simu_soc[ , 1]) == 0 & is.na(binn_obs_soc[ , 1]) == 0)
# Calculate the SOC NSE
current_data = cbind(as.vector(binn_simu_soc[valid_soc_loc, ]), as.vector(binn_obs_soc[valid_soc_loc, ]))/1000
current_data = data.frame(current_data[which(is.na(current_data[ , 1]) == 0), ])
colnames(current_data) = c('binn', 'proda')
# Calculate the SOC NSE
nse_process_middle = 1 - sum((current_data$binn - current_data$proda)^2)/sum((mean(current_data$proda) - current_data$proda)^2)

# Make the scatter plot
p_soc = make_scatter_plot(
    current_data, "SOC", nse_process_middle, NULL, NULL, TRUE)

p_para_performance_list[[plot_idx]] = p_soc
plot_idx = plot_idx + 1


# Third plot: box plot showing the mean correlation of the recovered parameters and the mean correlation of the SOC across the 10 cross-validation cases
# SOC NSE: cross_validation_soc_nse[ , 2]
# Recovered parameters correlation: cross_validation_para_mean_correlation
Mean_SOC_NSE = as.numeric(cross_validation_soc_nse[ , 2])
p_box_plot = ggplot() + 
    geom_boxplot(aes(x = 'SOC \n Mean NSE', y = Mean_SOC_NSE), color = '#2166AC', linewidth = 2, outlier.shape = 16,  outlier.size = 5)+
    geom_boxplot(aes(x = 'Parameters \n Mean Correlation', y = cross_validation_para_mean_correlation), color = '#B2182B', linewidth = 2, outlier.shape = 16) +
    xlab('') +
    ylab('') +
    ggtitle('Mean Performance') +
    theme_minimal(base_family = "Helvetica") +  
    # y axis range starts from 0
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
    # coord_flip() +
    theme(plot.background = element_blank()) +
    theme(axis.line = element_line(size = 1, color = 'black')) +
    theme(axis.text = element_text(size = 60, color = 'black')) +
    theme(axis.title = element_text(size = 60, color = 'black')) +
    theme(legend.position = 'None') +
    theme(plot.title = element_text(size = 80, hjust = 0.5)) +
    theme(plot.margin = unit(c(0.5, 0.5, 1, 0.2), 'inch'))

p_para_performance_list[[plot_idx]] = p_box_plot

# Factor the random seed for coloring
para_fold_plot$random_seed <- as.factor(para_fold_plot$random_seed)

# Plot the distribution of the parameters for all cross-validation cases (para_fold_plot)
p_para_density <- ggplot(para_fold_plot, aes(x = value, color = random_seed)) +
    geom_density(alpha = 0, size = 2) +
    facet_wrap(~ parameter, scales = "free", ncol = 4) +
    xlab('Parameter Value') +
    ylab('Density') +
    scale_x_continuous(limits = c(0, 1)) +
    theme_minimal(base_family = "Helvetica") +
    theme(legend.position = "None") +
    ggtitle("Parameter Distributions") + 
    theme(plot.margin = unit(c(0.2, 0, 0.2, 0), 'inch')) +
    theme(plot.title = element_text(size = 80, hjust = 0.5), strip.text = element_text(size = 60)) +
    theme(axis.text=element_text(size = 70, color = 'black'), axis.title = element_text(size = 80), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch'))


# Save the plots
# Arrange the plots with the shared legend on the left
# First, create a plot for the legend with appropriate sizing
legend_plot <- plot_grid(legend_plot, NULL, ncol = 1, rel_heights = c(1, 1))

# Create the main grid of plots
if (recovery_experiments[recovery_idx] == 'Five') {
  main_plots <- plot_grid(
      plotlist = p_para_performance_list[1:length(p_para_performance_list)-1], 
      ncol = plot_col,
      # add labels based on plot_idx
      labels = c(paste0("(", letters[1:(plot_idx-1)], ")")),
      label_size = 70,
      label_x = 0, 
      label_y = 1.1
  )
  box_plot <- plot_grid(
      p_para_performance_list[[plot_idx]],
      ncol = 1,
      labels = paste0("(", letters[plot_idx], ")"),
      label_size = 70,
      label_x = 0, 
      label_y = 1.1
  ) + theme(plot.margin = unit(c(5, 1, 5, 2), 'inch'))
  # Combine legend and main plots
  final_plot <- plot_grid(
      legend_plot, main_plots, box_plot,
      ncol = 3,
      rel_widths = c(0.06, 1, 0.5)
  )

} else {
  main_plots <- plot_grid(
      plotlist = p_para_performance_list, 
      ncol = plot_col,
      # add labels based on plot_idx
      labels = c(paste0("(", letters[1:(plot_idx)], ")")),
      label_size = 70, 
      label_x = 0, 
      label_y = 1.1
  )

  # Combine legend and main plots
  final_plot <- plot_grid(
      legend_plot, main_plots,
      ncol = 2,
      rel_widths = c(0.06, 1)
  )
}

final_plot <- final_plot + theme(plot.margin = unit(c(2, 0, 0, 0), 'inch'))

# Save the final plot
jpeg(paste(cross_validation_dir_output, '/', recovery_experiments[recovery_idx], '_test_para_scatter_plot.jpeg', sep = ''),
     width = plot_col_width, height = plot_col_height, units = 'in', res = 300)
print(final_plot)
dev.off()

jpeg(paste(cross_validation_dir_output, '/', recovery_experiments[recovery_idx], '_test_para_density_plot.jpeg', sep = ''),
     width = plot_col_width, height = plot_col_height, units = 'in', res = 300)
print(p_para_density)
dev.off()

print(paste0('Test Performance plot for ', recovery_experiments[recovery_idx], ' done'))


}
