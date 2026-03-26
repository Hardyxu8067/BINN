# Plot a box plot showing the NSE of BINN predicted SOC and observed SOC accross cross-validation folds
# Choose a cross-validation case with middle SOC recovery correlation
# Plot the map of the difference between the observed and predicted SOC
# Plot the scatter plot of the observed and predicted SOC
# Output one plot with map on top (a), scatter plot and box plot on the bottom (b and c)

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
library(patchwork)
library(usmap)
library(ggsignif)


##
rm(list = ls())

setwd('D:/Research/BINN/BINN_output/plot/')
# Sys.setenv(PROJ_LIB = "C:/Users/hx293/AppData/Local/R/win-library/4.3/sf/proj")
Sys.setenv(PROJ_LIB = "C:/Program Files/R/R-4.3.3/library/sf/proj")

## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
# diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#ffffff", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))
diff.colors <- colorRampPalette(c("#bf5700", "#f0954b", "#f9bd8c", "#dddddd", "#92C5DE", "#4393c3", "#2166ac"))

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
cross_validation_folder = 'Cross_Validation_PRODA_smaller_weight'
cross_validation_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/')
cross_validation_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/Output/')

# Get the list of all the folders in the cross validation directory
cross_validation_dir_list = list.dirs(cross_validation_dir_input, full.names = FALSE, recursive = FALSE)
# Exclude the folder of Output
cross_validation_dir_list = cross_validation_dir_list[!cross_validation_dir_list %in% 'Output']
# Get the test id by splitting the folder name by _ and select the second last element
PRODA_idx = sapply(strsplit(cross_validation_dir_list, '_'), function(x) x[length(x)])
PRODA_idx = sapply(strsplit(PRODA_idx, '='), function(x) x[2])
# change the type to integer
PRODA_idx = as.integer(PRODA_idx)

# PRODA data path
data_dir_PRODA = 'D:/Nutstore/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
data_dir_loc = 'D:/Nutstore/Research_Data/BINN/Server_Script/post_training/component_calculation/'
proda_para_input = 'D:/Research/Binn/PRODA_Results/'

# create output folder if not exist
if (!dir.exists(cross_validation_dir_output)) {
  dir.create(cross_validation_dir_output)
}

#############################################################################
# SOC NSE: PRODA vs BINN w/ same data sites vs BINN w/ different data sites
#############################################################################
BINN_soc_nse = array(NA, dim = c(length(PRODA_idx), 1))
BINN_soc_nse = cbind(cross_validation_dir_list, PRODA_idx, BINN_soc_nse)

PRODA_soc_nse = array(NA, dim = c(length(PRODA_idx), 1))
PRODA_soc_nse = cbind(cross_validation_dir_list, PRODA_idx, PRODA_soc_nse)

for (i in 1:length(cross_validation_dir_list)) {
    # Test SOC NSE for BINN with different data sites as PRODA
    binn_simu_soc = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Test/nn_test_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc = data.matrix(binn_simu_soc)
    binn_obs_soc = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/nn_obs_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_obs_soc = data.matrix(binn_obs_soc)
    valid_soc_loc_BINN = which(is.na(binn_simu_soc[ , 1]) == 0 & is.na(binn_obs_soc[ , 1]) == 0)
    # Calculate the SOC NSE
    middle_soc_corr_BINN = cbind(as.vector(binn_simu_soc[valid_soc_loc_BINN, ]), as.vector(binn_obs_soc[valid_soc_loc_BINN, ]))/1000
    middle_soc_corr_BINN = data.frame(middle_soc_corr_BINN[which(is.na(middle_soc_corr_BINN[ , 1]) == 0), ])
    colnames(middle_soc_corr_BINN) = c('binn', 'obs')
    BINN_soc_nse[i, 3] = 1 - sum((middle_soc_corr_BINN$binn - middle_soc_corr_BINN$obs)^2)/sum((mean(middle_soc_corr_BINN$obs) - middle_soc_corr_BINN$obs)^2)

    # Test SOC NSE for PRODA
    proda_simu_soc = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Test/PRODA_test_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    proda_simu_soc = data.matrix(proda_simu_soc)
    proda_obs_soc = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/nn_obs_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    proda_obs_soc = data.matrix(proda_obs_soc)
    valid_soc_loc_proda = which(is.na(proda_simu_soc[ , 1]) == 0 & is.na(proda_obs_soc[ , 1]) == 0)
    # Calculate the SOC NSE
    middle_soc_corr_proda = cbind(as.vector(proda_simu_soc[valid_soc_loc_proda, ]), as.vector(proda_obs_soc[valid_soc_loc_proda, ]))/1000
    middle_soc_corr_proda = data.frame(middle_soc_corr_proda[which(is.na(middle_soc_corr_proda[ , 1]) == 0), ])
    colnames(middle_soc_corr_proda) = c('binn', 'obs')
    PRODA_soc_nse[i, 3] = 1 - sum((middle_soc_corr_proda$binn - middle_soc_corr_proda$obs)^2)/sum((mean(middle_soc_corr_proda$obs) - middle_soc_corr_proda$obs)^2)
}

mean(as.numeric(BINN_soc_nse[ , 3]))
mean(as.numeric(PRODA_soc_nse[ , 3]))

# Create a new data frame to join the SOC NSE by idx
# Reorder each data frame by the idx
BINN_soc_nse = BINN_soc_nse[order(BINN_soc_nse[ , 2]), ]
PRODA_soc_nse_reordered = PRODA_soc_nse[order(PRODA_soc_nse[ , 2]), ]
cross_validation_soc_nse = data.frame(idx = as.numeric(BINN_soc_nse[ , 2]), 
                                       BINN_soc_nse = as.numeric(BINN_soc_nse[ , 3]), 
                                       PRODA_soc_nse = as.numeric(PRODA_soc_nse_reordered[ , 3]))


# Put the data in long format
cross_validation_soc_nse_long <- cross_validation_soc_nse %>% 
  pivot_longer(
    cols = c(BINN_soc_nse, PRODA_soc_nse),
    names_to  = "metric",
    values_to = "nse"
  )

# Test significance between BINN and PRODA using t-test
t_test_result = t.test(cross_validation_soc_nse$BINN_soc_nse, cross_validation_soc_nse$PRODA_soc_nse, paired = TRUE)
t_test_annotation = t_test_result$p.value

# Plot the box plot grouped by metric
mean_box_plot = ggplot(cross_validation_soc_nse_long, aes(x = metric, y = nse)) +
  geom_boxplot(linewidth = 2, outlier.shape = 16, width = 0.3, outlier.size = 8, outlier.color = 'red', outlier.fill = 'red') +
  geom_signif(comparisons = list(c('BINN_soc_nse', 'PRODA_soc_nse')), map_signif_level = TRUE, annotation = formatC(t_test_annotation, digits = 2), textsize = 13, tip_length = -0.2, y_position = 0.3, vjust = 2.5) +
  stat_summary(fun   = mean, geom  = "text", aes(label = sprintf("%.2f", ..y..)), vjust = -2.5, fontface = "bold", size  = 15) +
  labs(x = ' ', y = ' ') +
  scale_x_discrete(labels = c('BINN_soc_nse' = 'BINN ', 
                              'PRODA_soc_nse' = 'PRODA')) +
  ggtitle('Mean SOC NSE Performance') +
  theme_minimal(base_family = "Helvetica") +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.2, 1.2)) +
  theme(plot.background = element_blank()) +
  theme(axis.line = element_line(size = 1, color = 'black')) +
  theme(axis.text = element_text(size = 40, color = 'black')) +
  theme(axis.title = element_text(size = 55, color = 'black')) +
  theme(legend.position = 'None') +
  theme(plot.title = element_text(size = 55, hjust = 0.5)) +
  theme(plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), 'inch')) +
  theme(axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.25, 'inch')) 

jpeg(paste(cross_validation_dir_output, 'Mean_SOC_NSE_Performance.jpg', sep = ''), width = 20, height = 20, units = 'in', res = 300)
print(mean_box_plot)
dev.off()

#################################################################################
# Plot the difference of SOC in three different experiments
#################################################################################
world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/cb_2018_us_state_500k/cb_2018_us_state_500k.shp', layer = 'cb_2018_us_state_500k')
# coord_info = '+proj=robin'
coord_info = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
world_coastline <- st_transform(world_coastline, CRS(coord_info))

ocean_left = cbind(rep(-180, 100), seq(from = 80, to = -56, by = -(80 + 56)/(100 -1)))
ocean_right = cbind(rep(180, 100), seq(from = -56, to = 80, by = (80 + 56)/(100 -1)))
ocean_top = cbind(seq(from = 180, to = -180, by = -(360)/(100 -1)), rep(80, 100))
ocean_bottom = cbind(seq(from = -180, to = 180, by = (360)/(100 -1)), rep(-56, 100))

# Try to plot only the mainland US
US_left = cbind(rep(-180, 100), seq(from = 24, to = 50, by = -(24 - 50)/(100 -1)))
US_right = cbind(rep(180, 100), seq(from = 50, to = 24, by = (24 - 50)/(100 -1)))
US_top = cbind(seq(from = 180, to = -180, by = -(360)/(100 -1)), rep(50, 100))
US_bottom = cbind(seq(from = -180, to = 180, by = (360)/(100 -1)), rep(24, 100))

# world_ocean = rbind(ocean_left, ocean_bottom, ocean_right, ocean_top)
world_ocean = rbind(US_left, US_bottom, US_right, US_top)
world_ocean = as.matrix(world_ocean)

world_ocean <- project(xy = world_ocean, proj = coord_info)

world_ocean = data.frame(world_ocean)
colnames(world_ocean) = c('lon', 'lat')

# lat_limits = rbind(c(-62, 24.5), c(-140, 50))
# lat_limits = rbind(c(0, -56), c(0, 80))
lat_limits = rbind(c(-75, 21), c(-130, 48))
# lat_limits_robin = project(xy = as.matrix(lat_limits), proj = coord_info) 
lat_limits_albers = project(xy = as.matrix(lat_limits), proj = coord_info)

############################################################################
# Plot the difference of SOC in three different experiments
############################################################################
# For each cross validation case, plot the difference of SOC between the observed and predicted SOC
# And finally plot the overall difference of SOC accross all cross validation cases
# Initialize the data frame to store the difference of SOC
diff_soc_binn = data.frame()
diff_soc_PRODA = data.frame()

# Loop through each cross validation case
for (icase in 1:length(cross_validation_dir_list)) {

  selected_cross_validation_case_binn= cross_validation_dir_list[icase]

  # Read in the data
  pred_soc_binn = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_best_simu_soc_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  pred_soc_PRODA = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/PRODA_test_soc_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')

  # Observed SOC
  obs_soc_binn = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/nn_obs_soc_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  obs_soc_PRODA = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/nn_obs_soc_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')

  # Read in the depth data
  soc_upper_depth_binn = read.csv(paste(cross_validation_dir_input,  selected_cross_validation_case_binn, '/Test/nn_test_upper_depth_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  soc_upper_depth_PRODA = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_upper_depth_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  soc_lower_depth_binn = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_lower_depth_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  soc_lower_depth_PRODA = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_lower_depth_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')

  # Read in the lon and lat data
  binn_lon = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_lons_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  binn_lat = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_lats_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  PRODA_lon = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_lons_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')
  PRODA_lat = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case_binn, '/Test/nn_test_lats_', selected_cross_validation_case_binn, '.csv', sep = ''), header = TRUE, sep = ',')

  # Select rows where there's at least one valid soc prediction (among 200 columns)
  valid_profile_row = which(rowSums(!is.na(pred_soc_binn)) > 0 & rowSums(!is.na(pred_soc_PRODA)) > 0)

  # For each row, if valid_profile_loc is valid in that row, calculate the difference between the observation and prediction and times the difference between the upper and lower depth
  # If the valid_profile_loc is not valid in that row, set the difference to be NA
  # Store the sum of the difference in a new vector
  soc_diff_binn = rep(NA, nrow(pred_soc_binn))
  soc_diff_PRODA = rep(NA, nrow(pred_soc_PRODA))

  for (i in valid_profile_row) {
    temp_soc_diff_sum_binn = 0
    temp_soc_diff_sum_PRODA = 0

    for (j in 1:200) {
      if (!is.na(pred_soc_binn[i, j])) {
        temp_soc_diff_sum_binn = temp_soc_diff_sum_binn + (obs_soc_binn[i, j] - pred_soc_binn[i, j]) * (soc_lower_depth_binn[i, j] - soc_upper_depth_binn[i, j])
      }
      if (!is.na(pred_soc_PRODA[i, j])) {
        temp_soc_diff_sum_PRODA = temp_soc_diff_sum_PRODA + (obs_soc_PRODA[i, j] - pred_soc_PRODA[i, j]) * (soc_lower_depth_PRODA[i, j] - soc_upper_depth_PRODA[i, j])
      }
    }
    soc_diff_binn[i] = temp_soc_diff_sum_binn
    soc_diff_PRODA[i] = temp_soc_diff_sum_PRODA
  }


  # Get the lon and lat for plotting
  current_data_binn = cbind(binn_lon, binn_lat)
  colnames(current_data_binn) = c('lon', 'lat')
  current_data_PRODA = cbind(PRODA_lon, PRODA_lat)
  colnames(current_data_PRODA) = c('lon', 'lat')

  # Bind the lon and lat with the soc_diff
  current_data_binn = cbind(current_data_binn, soc_diff_binn)
  colnames(current_data_binn) = c('lon', 'lat', 'soc_diff')
  current_data_PRODA = cbind(current_data_PRODA, soc_diff_PRODA)
  colnames(current_data_PRODA) = c('lon', 'lat', 'soc_diff')

  # exclude the data with nan value for all input variables
  current_data_binn = current_data_binn[valid_profile_row, ]
  current_data_PRODA = current_data_PRODA[valid_profile_row, ]

  # Bind the data to the overall data frame
  diff_soc_binn = rbind(diff_soc_binn, current_data_binn)
  diff_soc_PRODA = rbind(diff_soc_PRODA, current_data_PRODA)


  # Start plotting the difference of SOC
  # transfer lon and lat to robinson projection 
  lon_lat_transfer_binn = project(xy = as.matrix(current_data_binn[ , c('lon', 'lat')]), proj = coord_info) 
  current_data_binn[ , c('lon', 'lat')] = lon_lat_transfer_binn
  lon_lat_transfer_PRODA = project(xy = as.matrix(current_data_PRODA[ , c('lon', 'lat')]), proj = coord_info)
  current_data_PRODA[ , c('lon', 'lat')] = lon_lat_transfer_PRODA
  # plot data only within the shapefile constraint
  current_data_binn_us <- st_as_sf(current_data_binn, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
  current_data_binn_us <- st_intersection(current_data_binn_us, world_coastline)
  current_data_PRODA_us <- st_as_sf(current_data_PRODA, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
  current_data_PRODA_us <- st_intersection(current_data_PRODA_us, world_coastline)

  # Extract the coordinates from the geometry column
  coords_binn <- st_coordinates(current_data_binn_us$geometry)
  coords_PRODA <- st_coordinates(current_data_PRODA_us$geometry)
  # Add lon and lat back to the data frame to the first two columns
  current_data_binn_us$lat <- coords_binn[, 2]
  current_data_binn_us$lon <- coords_binn[, 1]
  current_data_PRODA_us$lat <- coords_PRODA[, 2]
  current_data_PRODA_us$lon <- coords_PRODA[, 1]
  # move the longtitude to the first column
  current_data_binn_us <- current_data_binn_us[c("lon", "lat", setdiff(names(current_data_binn_us), c("lon", "lat")))]
  current_data_PRODA_us <- current_data_PRODA_us[c("lon", "lat", setdiff(names(current_data_PRODA_us), c("lon", "lat")))]
  # remove the geometry column
  current_data_binn_us <- st_drop_geometry(current_data_binn_us)
  current_data_PRODA_us <- st_drop_geometry(current_data_PRODA_us)
  # remove all column after the 3rd column
  current_data_binn_us <- current_data_binn_us[ , 1:3]
  current_data_PRODA_us <- current_data_PRODA_us[ , 1:3]

  # Normalized the difference between -1 and 1
  max_diff = max(current_data_binn_us$soc_diff, current_data_PRODA_us$soc_diff, na.rm = TRUE)
  min_diff = min(current_data_binn_us$soc_diff, current_data_PRODA_us$soc_diff, na.rm = TRUE)
  # If normalized by 95th percentile, use the following line
  max_diff = max(quantile(current_data_binn_us$soc_diff, 0.99, na.rm = TRUE), 
                 quantile(current_data_PRODA_us$soc_diff, 0.99, na.rm = TRUE)
                 , na.rm = TRUE)
  min_diff = min(quantile(current_data_binn_us$soc_diff, 0.01, na.rm = TRUE),
                 quantile(current_data_PRODA_us$soc_diff, 0.01, na.rm = TRUE)
                 , na.rm = TRUE)
  # For positive difference, normalized to 0 to 1
  for (i in 1:nrow(current_data_binn_us)) {
    if (current_data_binn_us$soc_diff[i] > 0) {
      current_data_binn_us$normalized_soc_diff[i] = current_data_binn_us$soc_diff[i] / max_diff
    } else {
      current_data_binn_us$normalized_soc_diff[i] = -1 * current_data_binn_us$soc_diff[i] / min_diff
    }
    if (current_data_PRODA_us$soc_diff[i] > 0) {
      current_data_PRODA_us$normalized_soc_diff[i] = current_data_PRODA_us$soc_diff[i] / max_diff
    } else {
      current_data_PRODA_us$normalized_soc_diff[i] = -1 * current_data_PRODA_us$soc_diff[i] / min_diff
    }
  }

  # Set all difference to -1 if the difference is less than -1, all difference greater than 1 to 1
  current_data_binn_us$normalized_soc_diff[current_data_binn_us$normalized_soc_diff < -1] = -1
  current_data_binn_us$normalized_soc_diff[current_data_binn_us$normalized_soc_diff > 1] = 1
  current_data_PRODA_us$normalized_soc_diff[current_data_PRODA_us$normalized_soc_diff < -1] = -1
  current_data_PRODA_us$normalized_soc_diff[current_data_PRODA_us$normalized_soc_diff > 1] = 1

  # Plot the difference of SOC
  map_diff_soc_binn = ggplot() +
	# geom_tile(data = current_data_binn_us, aes(x = lon, y = lat, fill = normalized_soc_diff), height = 60000, width = 60000, na.rm = TRUE) +
	geom_point(data = current_data_binn_us, aes(x = lon, y = lat, color = normalized_soc_diff), size = 6) +
	# scale_fill_gradientn(name = 'SOC Difference', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
	# Use diff.colors for the colorbar
	scale_color_gradientn(name = 'SOC\nDifference', colours = diff.colors(15), na.value="transparent", limits = c(-1, 1), trans = 'identity', oob = scales::squish) +
	geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
	# geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
	# coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
  # if using Albers projection
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
	# change the background to black and white
	# coord_equal() +
	ylim(lat_limits_albers[ , 2]) +
	# change the legend properties
	# theme(legend.position = 'none') +
	# theme(legend.justification = c(0, 0), legend.position = c(-0.1, 0.5), legend.background = element_rect(fill = NA), legend.text.align = 0, legend.key.height = unit(1.2, 'cm'), legend.key.width = unit(1, 'cm')) +
	theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
	# change the size of colorbar
	# guides(color = guide_colorbar(direction = 'vertical', barwidth = 4, barheight = 16, title = 'SOC\nDifference', title.position = 'top', title.hjust = 0, title.vjust = 2, frame.linewidth = 0), reverse = FALSE) +
  guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
  # theme(legend.text = element_text(size = 50, ), legend.title = element_text(size = 0)) +
  theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
	# add title
	labs(title = paste('BINN in Test:', icase)) +
	# modify the position of title
	theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
	# modify the font size
	# theme(axis.title = element_text(size = 30)) + 
	theme(axis.title = element_blank()) +
	theme(panel.background = element_rect(fill = NA, colour = NA)) +
	# modify the margin
	theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
	theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))


  map_diff_soc_PRODA = ggplot() +
  # geom_tile(data = current_data_binn_us, aes(x = lon, y = lat, fill = normalized_soc_diff), height = 60000, width = 60000, na.rm = TRUE) +
  geom_point(data = current_data_PRODA_us, aes(x = lon, y = lat, color = normalized_soc_diff), size = 6) +
  # scale_fill_gradientn(name = 'SOC Difference', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
  scale_color_gradientn(name = 'SOC\nDifference', colours = diff.colors(15), na.value="transparent", limits = c(-1, 1), trans = 'identity', oob = scales::squish) +
  geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
  # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
  # coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
  # if using Albers projection
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
  # change the background to black and white
  # coord_equal() +
  ylim(lat_limits_albers[ , 2]) +
  # change the legend properties
  # theme(legend.position = 'none') +
  # theme(legend.justification = c(0, 0), legend.position = c(-0.1, 0.5), legend.background = element_rect(fill = NA), legend.text.align = 0, legend.key.height = unit(1.2, 'cm'), legend.key.width = unit(1, 'cm')) +
  theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
  # change the size of colorbar
  # guides(color = guide_colorbar(direction = 'vertical', barwidth = 4, barheight = 16, title = 'SOC\nDifference', title.position = 'top', title.hjust = 0, title.vjust = 2, frame.linewidth = 0), reverse = FALSE) +
  guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
  # theme(legend.text = element_text(size = 50, ), legend.title = element_text(size = 0)) +
  theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
  # add title
  labs(title = paste('PRODA in Test:', icase)) +
  # modify the position of title
  theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
  # modify the font size
  # theme(axis.title = element_text(size = 30)) + 
  theme(axis.title = element_blank()) +
  theme(panel.background = element_rect(fill = NA, colour = NA)) +
  # modify the margin
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))

  eval(parse(text = paste('map_diff_soc_PRODA', icase, ' = map_diff_soc_PRODA', sep = '')))
  eval(parse(text = paste('map_diff_soc_binn', icase, ' = map_diff_soc_binn', sep = '')))
}

# Output the plots
jpeg(paste(cross_validation_dir_output, 'Diff_SOC_Maps_Cross_Validation.jpeg', sep = ''), width = 25, height = 50, units = 'in', res = 300)
plot_grid(map_diff_soc_binn1, map_diff_soc_PRODA1, 
          map_diff_soc_binn2, map_diff_soc_PRODA2, 
          map_diff_soc_binn3, map_diff_soc_PRODA3,
          map_diff_soc_binn4, map_diff_soc_PRODA4,
          map_diff_soc_binn5, map_diff_soc_PRODA5,
          map_diff_soc_binn6, map_diff_soc_PRODA6,
          map_diff_soc_binn7,  map_diff_soc_PRODA7,
          map_diff_soc_binn8,  map_diff_soc_PRODA8,
          map_diff_soc_binn9,  map_diff_soc_PRODA9,
          map_diff_soc_binn10, map_diff_soc_PRODA10,
          nrow = 10, ncol = 2,
          rel_widths = c(3, 3, 3, 0.10),
          labels = c('a', 'b', 
                     'c', 'd', 
                     'e', 'f', 
                     'g', 'h', 
                     'i', 'j', 
                     'k', 'l', 
                     'm', 'n', 
                     'o', 'p', 
                     'q', 'r', 
                     's', 't'),
          label_size = 70,
          label_x = 0.05, label_y = 1.05,
          label_fontfamily = 'Arial',
          label_fontface = 'bold'
)
dev.off()


# Plot the overall difference of SOC for all cross validation cases
current_data_BINN = diff_soc_binn
colnames(current_data_BINN) = c('lon', 'lat', 'soc_diff')
current_data_PRODA = diff_soc_PRODA
colnames(current_data_PRODA) = c('lon', 'lat', 'soc_diff')

# lon and lat transfer to robinson projection
lon_lat_transfer_BINN = project(xy = as.matrix(current_data_BINN[ , c('lon', 'lat')]), proj = coord_info)
current_data_BINN[ , c('lon', 'lat')] = lon_lat_transfer_BINN
lon_lat_transfer_PRODA = project(xy = as.matrix(current_data_PRODA[ , c('lon', 'lat')]), proj = coord_info)
current_data_PRODA[ , c('lon', 'lat')] = lon_lat_transfer_PRODA

# remove the geometry column
current_data_BINN = st_as_sf(current_data_BINN, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_data_PRODA = st_as_sf(current_data_PRODA, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_data_BINN <- st_intersection(current_data_BINN, world_coastline)
current_data_PRODA <- st_intersection(current_data_PRODA, world_coastline)

# Extract the coordinates from the geometry column
coords_BINN <- st_coordinates(current_data_BINN$geometry)
coords_PRODA <- st_coordinates(current_data_PRODA$geometry)

# Add lon and lat back to the data frame to the first two columns
current_data_BINN$lat <- coords_BINN[, 2]
current_data_PRODA$lat <- coords_PRODA[, 2]
current_data_BINN$lon <- coords_BINN[, 1]
current_data_PRODA$lon <- coords_PRODA[, 1]
# move the longtitude to the first column
current_data_BINN <- current_data_BINN[c("lon", "lat", setdiff(names(current_data_BINN), c("lon", "lat")))]
current_data_PRODA <- current_data_PRODA[c("lon", "lat", setdiff(names(current_data_PRODA), c("lon", "lat")))]
# remove the geometry column
current_data_BINN <- st_drop_geometry(current_data_BINN)
current_data_PRODA <- st_drop_geometry(current_data_PRODA)
# remove all column after the 3rd column
current_data_BINN <- current_data_BINN[ , 1:3]
current_data_PRODA <- current_data_PRODA[ , 1:3]

# Normalized the difference between -1 and 1
max_diff = max(current_data_BINN$soc_diff, current_data_PRODA$soc_diff, na.rm = TRUE)
min_diff = min(current_data_BINN$soc_diff, current_data_PRODA$soc_diff, na.rm = TRUE)

# If normalized by 95th percentile, use the following line
max_diff = max(quantile(current_data_BINN$soc_diff, 0.99, na.rm = TRUE), 
               quantile(current_data_PRODA$soc_diff, 0.99, na.rm = TRUE)
               , na.rm = TRUE)
min_diff = min(quantile(current_data_BINN$soc_diff, 0.01, na.rm = TRUE),
               quantile(current_data_PRODA$soc_diff, 0.01, na.rm = TRUE)
               , na.rm = TRUE)
               
# For positive difference, normalized to 0 to 1
for (i in 1:nrow(current_data_BINN)) {
  if (current_data_BINN$soc_diff[i] > 0) {
    current_data_BINN$normalized_soc_diff[i] <- current_data_BINN$soc_diff[i] / max_diff
  } else {
    current_data_BINN$normalized_soc_diff[i] <- -1 * current_data_BINN$soc_diff[i] / min_diff
  }
  if (current_data_PRODA$soc_diff[i] > 0) {
    current_data_PRODA$normalized_soc_diff[i] <- current_data_PRODA$soc_diff[i] / max_diff
  } else {
    current_data_PRODA$normalized_soc_diff[i] <- -1 * current_data_PRODA$soc_diff[i] / min_diff
  }
}

# Set all difference to -1 if the difference is less than -1, all difference greater than 1 to 1
current_data_BINN$normalized_soc_diff[current_data_BINN$normalized_soc_diff < -1] = -1
current_data_BINN$normalized_soc_diff[current_data_BINN$normalized_soc_diff > 1] = 1
current_data_PRODA$normalized_soc_diff[current_data_PRODA$normalized_soc_diff < -1] = -1
current_data_PRODA$normalized_soc_diff[current_data_PRODA$normalized_soc_diff > 1] = 1

# Plot the overall difference of SOC
map_diff_soc_BINN = ggplot() +
  geom_point(data = current_data_BINN, aes(x = lon, y = lat, color = normalized_soc_diff), size = 6) +
  scale_color_gradientn(name = 'SOC\nDifference\n', colours = diff.colors(15), na.value="transparent", limits = c(-1, 1), trans = 'identity', oob = scales::squish) +
  geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) +
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
  ylim(lat_limits_albers[ , 2]) +
  theme(legend.justification = c(0, 0), legend.position = c(0, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
  theme(legend.position = "none") +
  guides(color = guide_colorbar(direction = 'vertical', barwidth = 3, barheight = 15)) +
  theme(legend.text = element_text(size = 40, ), legend.title = element_text(size = 40)) +
  labs(title = 'BINN') +
  theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 55)) +
  theme(axis.title = element_blank()) +
  theme(panel.background = element_rect(fill = NA, colour = NA)) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))

map_diff_soc_PRODA = ggplot() +
  geom_point(data = current_data_PRODA, aes(x = lon, y = lat, color = normalized_soc_diff), size = 6) +
  scale_color_gradientn(name = 'SOC\nDifference\n', colours = diff.colors(15), na.value="transparent", limits = c(-1, 1), trans = 'identity', oob = scales::squish) +
  geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) +
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
  ylim(lat_limits_albers[ , 2]) + 
  theme(legend.justification = c(0, 0), legend.position = c(-0.11, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
  guides(color = guide_colorbar(direction = 'vertical', barwidth = 3, barheight = 15)) +
  theme(legend.text = element_text(size = 40, ), legend.title = element_text(size = 40)) +
  labs(title = 'PRODA') +
  theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 55)) +
  theme(axis.title = element_blank()) +
  theme(panel.background = element_rect(fill = NA, colour = NA)) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))

# Output the overall difference of SOC
jpeg(paste(cross_validation_dir_output, 'Diff_SOC_Overall_Cross_Validation.jpeg', sep = ''), width = 36, height = 15, units = 'in', res = 300)
plot_grid(map_diff_soc_BINN, map_diff_soc_PRODA, 
          nrow = 1, ncol = 2,
          rel_widths = c(3, 3, 0.10),
          labels = c('a', 'b'),
          label_size = 70,
          label_x = 0.05, label_y = 0.95,
          label_fontfamily = 'Arial',
          label_fontface = 'bold'
)
dev.off()

# Plot the box plot of normalized_soc_diff of experiments
box_plot_data = data.frame(
  Method = rep(c('BINN', 'PRODA'), each = nrow(current_data_BINN)),
  Normalized_SOC_Diff = c(current_data_BINN$normalized_soc_diff, current_data_PRODA$normalized_soc_diff)
)

# Significantly different using t-test
t_test_result = t.test(current_data_BINN$normalized_soc_diff, current_data_PRODA$normalized_soc_diff, alternative = 'two.sided', var.equal = FALSE)
print(t_test_result)

box_plot_data$Method = factor(box_plot_data$Method, levels = c('BINN', 'PRODA'))

box_plot = ggplot(box_plot_data, aes(x = Method, y = Normalized_SOC_Diff)) +
  geom_boxplot(linewidth = 2, outlier.shape = 16, width = 0.3, outlier.size = 2, outlier.color = 'red', outlier.fill = 'red') +
  geom_signif(comparisons = list(c('BINN', 'PRODA')), map_signif_level = TRUE, tip_length = 0.03, textsize = 12, y_position = 1) +
  # scale_fill_manual(values = c('#FF5733', '#33FF57', '#3357FF')) +
  labs(title = 'Normalized SOC Difference', x = ' ', y = ' ') +
  theme_minimal(base_family = "Helvetica") +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.2, 1.2)) +
  theme(plot.background = element_blank()) +
  theme(axis.line = element_line(size = 1, color = 'black')) +
  theme(axis.text = element_text(size = 40, color = 'black')) +
  theme(axis.title = element_text(size = 55, color = 'black')) +
  theme(legend.position = 'None') +
  theme(plot.title = element_text(size = 55, hjust = 0.5)) +
  theme(plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), 'inch')) +
  theme(axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.25, 'inch')) 
# Save the box plot
jpeg(paste(cross_validation_dir_output, 'Box_Plot_Normalized_SOC_Diff_Cross_Validation.jpeg', sep = ''), width = 20, height = 20, units = 'in', res = 300)
print(box_plot)
dev.off()

# Combine these plots into one figure
top_row <- plot_grid(mean_box_plot, box_plot, ncol = 2, labels = c('(a)', '(b)'), label_size = 70, label_x = 0, label_y = 1)
bottom_row <- plot_grid(map_diff_soc_BINN, map_diff_soc_PRODA, ncol = 2, labels = c('(c)', '(d)'), label_size = 70, label_x = 0, label_y = 1)
combined_plot <- plot_grid(top_row, bottom_row, ncol = 1, rel_heights = c(1.2, 1))
# Save the combined plot
jpeg(paste(cross_validation_dir_output, 'Combined_Plot_Cross_Validation.jpeg', sep = ''), width = 40, height = 30, units = 'in', res = 300)
print(combined_plot)
dev.off()
