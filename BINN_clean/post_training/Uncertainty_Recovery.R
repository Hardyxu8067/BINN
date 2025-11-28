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
# job_id = '20250922-024012_REPRO_BINN_RETRIEVAL_3198433_lr=1e-02_seed=111'
# recovery_experiments = c('All', 'Sixteen', 'Four')
# recovery_experiments = recovery_experiments[3]

cross_validation_folder = 'Recovery_MC_Dropout_CV'
cross_validation_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/')
cross_validation_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/Output/')

recovery_experiments = list.dirs(cross_validation_dir_input, full.names = FALSE, recursive = FALSE)
recovery_experiments = recovery_experiments[which(recovery_experiments != 'Output')]

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
               'w_scaling', 'beta')

## Loop through each recovery experiment
recovery_experiments = 'Five'

if (recovery_experiments == 'All') {
  para_index = 1:21
  bar_plot_width = 75
  num_col = 4
} else if (recovery_experiments == 'Sixteen') {
  para_index = c(0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20)
  para_index = para_index + 1
  num_col = 4
} else if (recovery_experiments == 'Four') {
  para_index = c(3, 9, 14, 19)
  para_index = para_index + 1
  bar_plot_width = 30
  num_col = 4
} else if (recovery_experiments == 'Five') {
  para_index = c(3, 9, 14, 19, 20)
  para_index = para_index + 1
  bar_plot_width = 40
  num_col = 3
}

recovery_experiment_dir = list.dirs(paste0(cross_validation_dir_input, '/', recovery_experiments), full.names = FALSE, recursive = FALSE)

# Initialize array for storing the results
MC_dropout_inside_percentage_all = array(NA, dim = c(length(recovery_experiment_dir), length(para_index)))
MC_dropout_outside_percentage_all = array(NA, dim = c(length(recovery_experiment_dir), length(para_index)))

for (recovery_experiment_file in recovery_experiment_dir) {
  # Get the MC_dropout directories for the current recovery experiment
  MC_dropout_dir_list = list.files(paste0(cross_validation_dir_input, '/',recovery_experiments, '/', recovery_experiment_file, '/MC_Dropout'), full.names = FALSE, recursive = FALSE)
  MC_dropout_soc_dir_list = list.files(paste0(cross_validation_dir_input, '/',recovery_experiments, '/', recovery_experiment_file, '/MC_Dropout'), pattern = 'MC_test_simu_soc_', full.names = FALSE, recursive = FALSE)
  MC_dropout_para_dir_list = list.files(paste0(cross_validation_dir_input, '/',recovery_experiments, '/', recovery_experiment_file, '/MC_Dropout'), pattern = 'MC_test_pred_para_', full.names = FALSE, recursive = FALSE)
  # Get the list of random seeds (the last part of the folder name after =)
  MC_idx_list = sapply(strsplit(MC_dropout_dir_list, split = '_'), function(x) x[length(x)])
  # Get only the numeric part from MC_idx_list
  MC_idx_list = parse_number(MC_idx_list)
  MC_idx_list = as.numeric(MC_idx_list)
  MC_idx_list = unique(MC_idx_list)
  # Check if MC_idx_list is continuous
  if (!all(diff(sort(MC_idx_list)) == 1)) {
    stop('MC_idx_list is not continuous')
  } else {
    print(paste0('MC_idx_list is continuous for: ', recovery_experiment_file, ', from ', min(MC_idx_list), ' to ', max(MC_idx_list)))
  }

  #############################################################################
  # Cross Validation SOC NSE and Parameter Correlation
  #############################################################################
  MC_dropout_soc_nse = array(NA, dim = c(length(MC_dropout_soc_dir_list), 1))
  MC_dropout_soc_nse = cbind(MC_dropout_soc_dir_list, MC_dropout_soc_nse)
  MC_dropout_para_correlation = array(NA, dim = c(length(MC_dropout_para_dir_list), length(para_index)))
  MC_dropout_para_mean_correlation = array(NA, dim = c(length(MC_dropout_para_dir_list), 1))
  MC_dropout_para_correlation = cbind(MC_dropout_para_dir_list, MC_dropout_para_correlation)

  # Load the test pred para files to get the valid profile ids
  temp_test_para_file = read_csv(paste0(cross_validation_dir_input, '/',recovery_experiments, '/', recovery_experiment_file, '/Test/nn_test_best_pred_para_', recovery_experiment_file, '.csv'), col_names = FALSE)
  temp_test_para_file = data.matrix(temp_test_para_file)
  valid_profile_id = which(is.na(temp_test_para_file[,1]) == 0)

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

  # Import the data first
  MC_dropout_soc = array(NA, dim = c(length(MC_dropout_soc_dir_list), length(valid_profile_id), 200))
  MC_dropout_para = array(NA, dim = c(length(MC_dropout_para_dir_list), length(valid_profile_id), length(para_index)))

  for (j in 1:length(MC_idx_list)) {
      # Load the SOC file
      temp_soc_file = read_csv(paste0(cross_validation_dir_input,'/',recovery_experiments, '/', recovery_experiment_file, '/MC_Dropout/', MC_dropout_soc_dir_list[j]), col_names = FALSE)
      temp_soc_file = data.matrix(temp_soc_file)

      # Load the predicted parameter file
      temp_para_file = read_csv(paste0(cross_validation_dir_input,'/',recovery_experiments, '/', recovery_experiment_file, '/MC_Dropout/', MC_dropout_para_dir_list[j]), col_names = FALSE)
      temp_para_file = data.matrix(temp_para_file)

      # Store the soc site data
      MC_dropout_soc[j, , ] = temp_soc_file
      MC_dropout_para[j, , ] = temp_para_file[, para_index]

  }
  test_selected_para = read_csv(paste0(cross_validation_dir_input, '/',recovery_experiments, '/', recovery_experiment_file, '/Test/nn_test_best_pred_para_', recovery_experiment_file, '.csv'), col_names = FALSE)


  # Plot the posterior distribution of the parameters for each profile
  # Also check whether the PRODA parameter is within the distribution
  MC_dropout_para_inside = array(NA, dim = c(length(valid_profile_id), length(para_index)))

  # randomly select 100 profiles from valid_profile_id
  set.seed(111)
  sampled_profile_id = sample(valid_profile_id, 10)

  for (i in 1:length(valid_profile_id)) {
      MC_dropout_soc_site = MC_dropout_soc[ , i, ]
      MC_dropout_para_site = MC_dropout_para[ , i, ]

      # Test para
      temp_test_para = data.matrix(test_selected_para)[valid_profile_id[i], para_index]

      # Actual PRODA para
      proda_selected_para = PRODA_para[which(PRODA_para$profile_id == valid_profile_id[i]), 2:ncol(PRODA_para)]
      temp_proda_para = as.numeric(proda_selected_para[1, para_index])

      # Plot the distribution of each parameter, with the test para as red line and the PRODA para as blue line
      para_list = list()

      for (k in 1:length(para_index)) {
      para_name = para_names[para_index[k]]
      para_values = as.numeric(MC_dropout_para_site[, k])
      para_values = para_values[is.na(para_values) == 0]
      para_df = data.frame(para_values = para_values)

      MC_dropout_para_inside[i, k] = ifelse(temp_proda_para[k] >= min(para_values) & temp_proda_para[k] <= max(para_values), 1, 0)
      
      if (valid_profile_id[i] %in% sampled_profile_id & recovery_experiment_file == recovery_experiment_dir[1]) {
          p <- ggplot(para_df, aes(x = para_values)) +
              geom_histogram(aes(y = ..density..), bins = 30, fill = 'lightgrey', color = 'black') +
              geom_density(color = 'black', size = 2) +
              geom_vline(xintercept = temp_test_para[k], color = '#B2182B', size = 3, linetype = 'dashed') +
              geom_vline(xintercept = temp_proda_para[k], color = '#2166AC', size = 3, linetype = 'dashed') +
              labs(title = paste0(para_name), x = 'Value', y = 'Density') +
              theme_minimal() +
              scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
              theme(plot.title = element_text(hjust = 0.5, size = 80),
                  plot.margin = unit(c(0.2, 0.5, 0.8, 0.2), 'inch'),
                  axis.text = element_text(size = 70, color = 'black'),
                  axis.title = element_text(size = 80),
                  axis.line = element_line(size = 1),
                  axis.ticks = element_line(size = 1, color = 'black'),
                  axis.ticks.length = unit(0.12, 'inch'),
                  legend.position = "none")
      
          para_list[[length(para_list) + 1]] <- p
      }
      }

      if (valid_profile_id[i] %in% sampled_profile_id & recovery_experiment_file == recovery_experiment_dir[1]) {
          para_plot <- plot_grid(
              plotlist = para_list,
              ncol = num_col,
              labels = c(paste0("(", letters[1:(length(para_list))], ")")),
              label_size = 70, 
              label_x = 0, 
              label_y = 1
          )

          jpeg(paste0(cross_validation_dir_output, 'MC_Dropout_Para_Profile_', valid_profile_id[i], '.jpeg'), 
                  width = 60, height = 16*round((length(para_list))/num_col), units = 'in', res = 300)
          plot(para_plot)
          dev.off()
      }
  }

  # Plot a bar plot showing the percentage of PRODA parameters within the MC dropout predicted parameter distribution for each parameter
  MC_dropout_para_inside_percentage = colSums(MC_dropout_para_inside, na.rm = TRUE) / nrow(MC_dropout_para_inside) * 100
  MC_dropout_para_outside_percentage = 100 - MC_dropout_para_inside_percentage
  # Store the results
  MC_dropout_inside_percentage_all[which(recovery_experiment_dir == recovery_experiment_file), ] = MC_dropout_para_inside_percentage
  MC_dropout_outside_percentage_all[which(recovery_experiment_dir == recovery_experiment_file), ] = MC_dropout_para_outside_percentage

  if (recovery_experiment_file == recovery_experiment_dir[1]) {
    # Plot a box plot showing the correlation of the recovered parameters and the NSE of the SOC across the MC dropout runs
    MC_dropout_soc_nse = array(NA, dim = c(length(MC_dropout_soc_dir_list), 1))
    MC_dropout_para_corr = array(NA, dim = c(length(MC_dropout_para_dir_list), length(para_index)))

    binn_obs_soc = read_csv(paste0(cross_validation_dir_input, '/', recovery_experiments, '/', recovery_experiment_file, '/nn_obs_soc_', recovery_experiment_file, '.csv'), col_names = FALSE)
    binn_obs_soc = data.matrix(binn_obs_soc)

    test_binn_soc = read_csv(paste0(cross_validation_dir_input, '/', recovery_experiments, '/', recovery_experiment_file, '/Test/nn_test_best_simu_soc_', recovery_experiment_file, '.csv'), col_names = FALSE)
    test_binn_soc = data.matrix(test_binn_soc)

    for (m in 1:length(MC_idx_list)) {
      # NSE for SOC
      binn_simu_soc = MC_dropout_soc[m, , ]
      valid_soc_loc = which(is.na(binn_obs_soc[ ,1]) == 0 & is.na(test_binn_soc[ ,1]) == 0)
      middle_soc_corr = cbind(as.vector(binn_simu_soc), as.vector(binn_obs_soc[valid_soc_loc, ])) / 1000
      middle_soc_corr = data.frame(middle_soc_corr[which(is.na(middle_soc_corr[ , 1]) == 0), ])
      colnames(middle_soc_corr) = c('binn', 'obs')
      MC_dropout_soc_nse[m, 1] = 1- (sum((middle_soc_corr$binn - middle_soc_corr$obs)^2) / sum((middle_soc_corr$obs - mean(middle_soc_corr$obs))^2))

      # Correlation for parameters
      binn_pred_para = MC_dropout_para[m, , ]
      valid_para_loc_temp = which(is.na(test_selected_para[ ,1]) == 0)
      valid_para_loc = intersect(valid_para_loc_temp, PRODA_para$profile_id)
      # Calculate the parameter correlation
        for (ipara in 1:length(para_index)) {
            current_para = cbind(PRODA_para[PRODA_para$profile_id %in% valid_para_loc, para_index[ipara]+1], binn_pred_para[, ipara])
            current_para = data.frame(current_para)
            colnames(current_para) = c('proda', 'binn')
            MC_dropout_para_corr[m, ipara] = cor(current_para$proda, current_para$binn, method = 'pearson')
        }
    }

    # Plot the box plot of the parameter correlation and the SOC NSE
    colnames(MC_dropout_para_corr) = para_names[para_index]
    colnames(MC_dropout_soc_nse) = c('SOC_NSE')
    MC_dropout_para_corr_df = data.frame(MC_dropout_para_corr)
    MC_dropout_para_corr_df_long = pivot_longer(MC_dropout_para_corr_df, 
                                                cols = everything(),
                                                names_to = 'Parameter', 
                                                values_to = 'Correlation')
    MC_dropout_para_corr_df_long$Parameter = factor(MC_dropout_para_corr_df_long$Parameter, levels = para_names[para_index])

    p_box_plot <- ggplot() +
        geom_boxplot(aes(x = 'SOC \n Mean NSE', y = MC_dropout_soc_nse[,1]), color = '#2166AC', linewidth = 2, outlier.shape = 16,  outlier.size = 5)+
        # Plot the parameter correlation boxplots in red(#B2182B)
        geom_boxplot(data = MC_dropout_para_corr_df_long, aes(x = Parameter, y = Correlation), color = '#B2182B', linewidth = 2, outlier.shape = 16,  outlier.size = 5) +
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
        theme(axis.title = element_text(size = 80, color = 'black')) +
        theme(legend.position = 'None') +
        theme(plot.title = element_text(size = 80, hjust = 0.5)) +
        theme(plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), 'inch'))

    jpeg(paste0(cross_validation_dir_output, 'MC_Dropout_Para_SOC_Boxplot.jpeg'), width = bar_plot_width, height = 35, units = 'in', res = 300)
    plot(p_box_plot)
    dev.off()
  }
}

# Store the mean and std for each parameter
MC_dropout_para_inside_percentage_mean = colMeans(MC_dropout_inside_percentage_all, na.rm = TRUE)
MC_dropout_para_inside_percentage_sd = apply(MC_dropout_inside_percentage_all, 2, sd, na.rm = TRUE)
# Combine mean and sd into a data frame for plotting
MC_dropout_para_plot_df = data.frame(Parameter = para_names[para_index],
                                    Inside_Mean = MC_dropout_para_inside_percentage_mean,
                                    Inside_SD = MC_dropout_para_inside_percentage_sd)
MC_dropout_para_plot_df$Outside_Mean = 100 - MC_dropout_para_plot_df$Inside_Mean
# Long format for ggplot
MC_dropout_para_plot_df_long <- MC_dropout_para_plot_df %>%
  pivot_longer(cols = c('Inside_Mean', 'Outside_Mean'),
               names_to = 'Type',
               values_to = 'Percentage') %>%
  mutate(Type = factor(Type, levels = c('Outside_Mean', 'Inside_Mean'),
                       labels = c('Outside', 'Inside')))
                                            

# Stack the outside mean on top of the inside mean for bar plot
p_bar_plot <- ggplot(MC_dropout_para_plot_df_long, aes(x = Parameter, y = Percentage, fill = Type)) +
    geom_bar(stat = 'identity', position = 'stack', color = 'transparent', linewidth = 1, width = 0.7) +
    geom_errorbar(data = subset(MC_dropout_para_plot_df_long, Type == 'Inside'),
                  aes(x = Parameter, y = Percentage, ymin = Percentage - MC_dropout_para_inside_percentage_sd, ymax = Percentage + MC_dropout_para_inside_percentage_sd),
                  width = 0.2, color = 'black', linewidth = 1.5) +
    scale_fill_manual(values = c('Inside' = '#e7a516', 'Outside' = '#a6a6a6')) +
    xlab('') +
    ylab('Percentage (%)') +
    ggtitle(' ') +
    theme_minimal(base_family = "Helvetica") +  
    # y axis range starts from 0
    scale_y_continuous(expand = c(0, 0), limits = c(0, 100)) +
    theme(plot.background = element_blank()) +
    theme(axis.line = element_line(size = 1, color = 'black')) +
    theme(axis.text = element_text(size = 80, color = 'black')) +
    theme(axis.text.x = element_text(margin = margin(t = 20))) +
    theme(axis.title = element_text(size = 80, color = 'black')) +
    theme(legend.position = 'right') +
    theme(legend.text = element_text(size = 60)) +
    theme(legend.title = element_text(size = 0)) +
    guides(fill = guide_legend(override.aes = list(size=10))) +
    theme(plot.title = element_text(size = 80, hjust = 0.5)) +
    theme(plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), 'inch'))

jpeg(paste0(cross_validation_dir_output, 'MC_Dropout_Para_Recovery_Barplot.jpeg'), width = bar_plot_width, height = 25, units = 'in', res = 300)
plot(p_bar_plot)
dev.off()
