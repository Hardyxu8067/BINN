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

##
rm(list = ls())

setwd('D:/Research/BINN/BINN_output/plot/')
Sys.setenv(PROJ_LIB = "C:/Program Files/R/R-4.3.3/library/sf/proj")

## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
# diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "white", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))
# From Orange to Blue
diff.colors <- colorRampPalette(c("#bf5700", "#f0954b", "#f9bd8c", "#f6f6f6", "#92C5DE", "#4393c3", "#2166ac"))

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
cross_validation_folder = 'CV_No_Spatial_Info'
cross_validation_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/')
cross_validation_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/Output/')
# Get the list of all the folders in the cross validation directory
cross_validation_dir_list = list.dirs(cross_validation_dir_input, full.names = FALSE, recursive = FALSE)
# Exclude the folder of Output
cross_validation_dir_list = cross_validation_dir_list[!cross_validation_dir_list %in% 'Output']


# PRODA data path
data_dir_PRODA = 'D:/Nutstore/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
data_dir_loc = 'D:/Nutstore/hx293/Research_Data/BINN/Server_Script/post_training/component_calculation/'
proda_para_input = 'D:/Research/Binn/PRODA_Results/'

# create output folder if not exist
if (!dir.exists(cross_validation_dir_output)) {
  dir.create(cross_validation_dir_output)
}

#############################################################################
# Cross Validation SOC NSE 
#############################################################################
cross_validation_soc_nse = array(NA, dim = c(length(cross_validation_dir_list), 1))
cross_validation_soc_nse = cbind(cross_validation_dir_list, cross_validation_soc_nse)
max_sample_num = 0
max_profile_num = 0

for (i in 1:length(cross_validation_dir_list)) {
    # Test SOC
    binn_simu_soc = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Test/nn_test_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc = data.matrix(binn_simu_soc)
    binn_simu_soc_train = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Train/nn_train_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc_train = data.matrix(binn_simu_soc_train)
    binn_simu_soc_val = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Validation/nn_val_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc_val = data.matrix(binn_simu_soc_val)
    binn_obs_soc = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/nn_obs_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_obs_soc = data.matrix(binn_obs_soc)
    valid_soc_loc = which(is.na(binn_simu_soc[ , 1]) == 0 & is.na(binn_obs_soc[ , 1]) == 0)
    # Calculate the SOC NSE
    middle_soc_corr = cbind(as.vector(binn_simu_soc[valid_soc_loc, ]), as.vector(binn_obs_soc[valid_soc_loc, ]))/1000
    middle_soc_corr = data.frame(middle_soc_corr[which(is.na(middle_soc_corr[ , 1]) == 0), ])
    colnames(middle_soc_corr) = c('binn', 'obs')
    cross_validation_soc_nse[i, 2] = 1 - sum((middle_soc_corr$binn - middle_soc_corr$obs)^2)/sum((mean(middle_soc_corr$obs) - middle_soc_corr$obs)^2)
    # Update the max sample number
    # Check the number of valid samples for each row and get the maximum number of valid samples
    binn_simu_soc_all = binn_simu_soc_train
    nan_loc = which(is.na(binn_simu_soc_all) == 1)
    binn_simu_soc_all[nan_loc] = binn_simu_soc_val[nan_loc]
    nan_loc = which(is.na(binn_simu_soc_all) == 1)
    binn_simu_soc_all[nan_loc] = binn_simu_soc[nan_loc]

    temp_profile_num = sum(!is.na(binn_simu_soc_all[,1]))
    temp_max = max(rowSums(!is.na(binn_simu_soc_all)))

    max_sample_num = max(max_sample_num, temp_max)
    max_profile_num = max(max_profile_num, temp_profile_num)
}

mean(as.numeric(cross_validation_soc_nse[ , 2]))

# Select the cross-validation case with the middle SOC recovery correlation
# Sort the cross-validation cases by the SOC NSE
selected_cross_validation_case = cross_validation_soc_nse[order(cross_validation_soc_nse[ , 2], decreasing = TRUE), 1][5]

##############################################
# Begin plotting
##############################################
# Plot the uncertainty distribution of SOC
cross_validation_binn_soc = array(NA, dim = c(length(cross_validation_dir_list), max_profile_num, max_sample_num))
cross_validation_obs_soc = binn_obs_soc
cross_validation_obs_soc = cross_validation_obs_soc[!is.na(cross_validation_obs_soc[,1]), ]
if (nrow(cross_validation_obs_soc) != max_profile_num) {
    stop('Number of profiles in observed SOC is not equal to max_profile_num')
}

for (i in 1:length(cross_validation_dir_list)) {
    # Test SOC
    binn_simu_soc_train = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Train/nn_train_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc_train = data.matrix(binn_simu_soc_train)
    binn_simu_soc_val = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Validation/nn_val_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc_val = data.matrix(binn_simu_soc_val)
    binn_simu_soc_test = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[i], '/Test/nn_test_best_simu_soc_', cross_validation_dir_list[i], '.csv', sep = ''), header = FALSE, sep = ',')
    binn_simu_soc_test = data.matrix(binn_simu_soc_test)
    # Combine the train, val, and test data
    binn_simu_soc = binn_simu_soc_train
    nan_loc = which(is.na(binn_simu_soc) == 1)
    binn_simu_soc[nan_loc] = binn_simu_soc_val[nan_loc]
    nan_loc = which(is.na(binn_simu_soc) == 1)
    binn_simu_soc[nan_loc] = binn_simu_soc_test[nan_loc]
    cross_validation_binn_soc[i, , ]= binn_simu_soc[!is.na(binn_simu_soc[,1]), 1:max_sample_num]
}

# Calculate the mean and standard deviation of the SOC predictions across cross-validation folds
mean_cross_validation_binn_soc = matrix(NA, nrow = max_profile_num, ncol = max_sample_num)
sd_cross_validation_binn_soc = matrix(NA, nrow = max_profile_num, ncol = max_sample_num)
for (i in 1:max_profile_num) {
    for (j in 1:max_sample_num) {
        mean_cross_validation_binn_soc[i, j] = mean(cross_validation_binn_soc[ , i, j], na.rm = TRUE)
        sd_cross_validation_binn_soc[i, j] = sd(cross_validation_binn_soc[ , i, j], na.rm = TRUE)
    }
}

###########################################
# Plot the selected cross-validation case #
###########################################
# Initialize the plot
p_soc_list = list()
plot_idx = 1

## First plot the map of the difference between the observed and predicted SOC
# Read in data
binn_pred_soc = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_best_simu_soc_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_pred_soc = data.matrix(binn_pred_soc)
binn_obs_soc = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/nn_obs_soc_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_soc = data.matrix(binn_obs_soc)

## Depth Data for each soc observation
soc_upper_depth = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_upper_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_upper_depth = data.matrix(soc_upper_depth)
soc_lower_depth = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_lower_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_lower_depth = data.matrix(soc_lower_depth)

# Select rows where there's at least one valid soc prediction (among 200 columns)
valid_profile_row = which(rowSums(!is.na(binn_pred_soc)) > 0)

# For each row, if valid_profile_loc is valid in that row, calculate the difference between the observation and prediction and times the difference between the upper and lower depth
# If the valid_profile_loc is not valid in that row, set the difference to be NA
# Store the sum of the difference in a new vector
soc_diff = rep(NA, nrow(binn_pred_soc))
for (i in valid_profile_row) {
	temp_soc_diff_sum = 0
	for (j in 1:200) {
		if (!is.na(binn_pred_soc[i, j])) {
			temp_soc_diff_sum = temp_soc_diff_sum + (binn_obs_soc[i] - binn_pred_soc[i, j]) * (soc_lower_depth[i, j] - soc_upper_depth[i, j])
		}
	}
	soc_diff[i] = temp_soc_diff_sum
}

# Get the lon and lat for plotting
binn_lon = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_lons_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lat = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_lats_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
current_data_binn = cbind(binn_lon, binn_lat)
colnames(current_data_binn) = c('lon', 'lat')

# Bind the lon and lat with the soc_diff
current_data_binn = cbind(current_data_binn, soc_diff)
colnames(current_data_binn) = c('lon', 'lat', 'soc_diff')

# exclude the data with nan value for all input variables
current_data_binn = current_data_binn[valid_profile_row, ]


#-------------------------------------soc stock and Residence Time map
world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/cb_2018_us_state_500k/cb_2018_us_state_500k.shp', layer = 'cb_2018_us_state_500k')
# world_coastline <- us_map(regions = "states")
# Define the bounding box for the mainland U.S. (excluding Alaska)
# coord_info = '+proj=robin'
# Albers projection
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
lat_limits = rbind(c(-75, 21), c(-130, 48))
# if using Albers projection
# lat_limits = rbind(c(24.5, -62), c(50, -140))
# lat_limits = rbind(c(0, -56), c(0, 80))
# lat_limits_robin = project(xy = as.matrix(lat_limits), proj = coord_info) 
lat_limits_albers = project(xy = as.matrix(lat_limits), proj = coord_info)

# transfer lon and lat to robinson projection 
lon_lat_transfer = project(xy = as.matrix(current_data_binn[ , c('lon', 'lat')]), proj = coord_info) 
current_data_binn[ , c('lon', 'lat')] = lon_lat_transfer
# plot data only within the shapefile constraint
current_data_binn_us <- st_as_sf(current_data_binn, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_data_binn_us <- st_intersection(current_data_binn_us, world_coastline)

# Extract the coordinates from the geometry column
coords <- st_coordinates(current_data_binn_us$geometry)
# Add lon and lat back to the data frame to the first two columns
current_data_binn_us$lat <- coords[, 2]
current_data_binn_us$lon <- coords[, 1]
# move the longtitude to the first column
current_data_binn_us <- current_data_binn_us[c("lon", "lat", setdiff(names(current_data_binn_us), c("lon", "lat")))]
# remove the geometry column
current_data_binn_us <- st_drop_geometry(current_data_binn_us)
# remove all column after the 3rd column
current_data_binn_us <- current_data_binn_us[ , 1:3]

# Normalized the difference between -1 and 1
max_soc = max(quantile(current_data_binn_us$soc_diff, 0.99, na.rm = TRUE), na.rm = TRUE)
min_soc = min(quantile(current_data_binn_us$soc_diff, 0.01, na.rm = TRUE), na.rm = TRUE)
# For positive difference, normalized to 0 to 1
for (i in 1:nrow(current_data_binn_us)) {
	if (current_data_binn_us$soc_diff[i] > 0) {
		current_data_binn_us$normalized_soc_diff[i] = current_data_binn_us$soc_diff[i] / max_soc
	} else {
		current_data_binn_us$normalized_soc_diff[i] = -1 * current_data_binn_us$soc_diff[i] / min_soc
	}
}

# Set all difference to -1 if the difference is less than -1, all difference greater than 1 to 1
current_data_binn_us$normalized_soc_diff[current_data_binn_us$normalized_soc_diff < -1] = -1
current_data_binn_us$normalized_soc_diff[current_data_binn_us$normalized_soc_diff > 1] = 1

# Set the lower and upper limit for the legend
legend_lower_diff_soc = min(current_data_binn_us$normalized_soc_diff, na.rm = TRUE)
legend_upper_diff_soc = max(current_data_binn_us$normalized_soc_diff, na.rm = TRUE)

# Plot the difference map for SOC
map_diff_soc = ggplot() +
	# geom_tile(data = current_data_binn_us, aes(x = lon, y = lat, fill = normalized_soc_diff), height = 60000, width = 60000, na.rm = TRUE) +
	geom_point(data = current_data_binn_us, aes(x = lon, y = lat, color = normalized_soc_diff), size = 8, alpha = 0.8) +
	# scale_fill_gradientn(name = 'SOC Difference', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
	# Use diff.colors for the colorbar
	scale_color_gradientn(name = 'SOC Difference', colours = diff.colors(15), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
	geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 0.6) + 
	# geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
	# coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
  # if using Albers projection
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
	# change the background to black and white
	# coord_equal() +
	# theme_map() +
	# ylim(lat_limits_albers[ , 2]) +
	# change the legend properties
	# theme(legend.position = 'none') +
	theme(legend.justification = c(0, 0), legend.position = c(-0.1, 0.5), legend.background = element_rect(fill = NA), legend.text.align = 0, legend.key.height = unit(1.2, 'cm'), legend.key.width = unit(1, 'cm')) +
	# theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
	# change the size of colorbar
	guides(color = guide_colorbar(direction = 'vertical', barwidth = 4, barheight = 16, title = 'SOC\nDifference', title.position = 'top', title.hjust = 0, title.vjust = 2, frame.linewidth = 0), reverse = FALSE) +
  theme(legend.text = element_text(size = 50, ), legend.title = element_text(size = 0)) +
	# add title
	labs(title = ' ') +
	# modify the position of title
	theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 55)) + 
	# modify the font size
	# theme(axis.title = element_text(size = 30)) + 
	theme(axis.title = element_blank()) +
	theme(panel.background = element_rect(fill = NA, colour = NA)) +
	# modify the margin
	theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
	theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))
	# theme(axis.text=element_text(size = 15, color = 'black'))

p_soc_list[[plot_idx]] = map_diff_soc
plot_idx = plot_idx + 1

## Second plot the scatter plot of the observed and predicted SOC
binn_simu_soc = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_best_simu_soc_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_simu_soc = data.matrix(binn_simu_soc)
binn_obs_soc = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/nn_obs_soc_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_obs_soc = data.matrix(binn_obs_soc)
valid_soc_loc = which(is.na(binn_simu_soc[ , 1]) == 0 & is.na(binn_obs_soc[ , 1]) == 0)
# Calculate the SOC NSE
current_data = cbind(as.vector(binn_simu_soc[valid_soc_loc, ]), as.vector(binn_obs_soc[valid_soc_loc, ]))/1000
current_data = data.frame(current_data[which(is.na(current_data[ , 1]) == 0), ])
colnames(current_data) = c('binn', 'obs')
# Calculate the SOC NSE
nse_process_middle = 1 - sum((current_data$binn - current_data$obs)^2)/sum((mean(current_data$obs) - current_data$obs)^2)

p_soc = ggplot(data = current_data) + 
    stat_bin_hex(aes(x = obs, y = binn), bins = 100) +
    scale_fill_gradientn(name = 'Count', colors = viridis(7), trans = 'identity', limits = c(1, 20), oob = scales::squish) +
    scale_y_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
    scale_x_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
    geom_abline(slope = 1, intercept = 0, size = 1, color = 'black') +
    # Add NSE text
    annotate("text", x = 0.1, y = 360, label = paste("NSE:", round(nse_process_middle, 2)), size = 12, hjust = 0.15, vjust = 0) +
    theme_classic() + 
    # add title
    labs(title = paste(' ', sep = ''),
    x = expression(paste('Observed SOC (kg C m'^'-3', ')', sep = '')), y = expression(paste('BINN simulation (kg C m'^'-3', ')', sep = ''))) +
    # change the legend properties
    guides(fill = guide_colorbar(direction = 'horizontal', barwidth = 15, barheight = 2.5, title.position = 'right', title.hjust = 0, title.vjust = 0.8, label.hjust = 0.5, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 35), legend.title = element_text(size = 35))  +
    theme(legend.justification = c(0, 0), legend.position = c(0, 0.9), legend.background = element_rect(fill = NA)) + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, size = 55)) + 
    # modify the font size
    # modify the margin
    # theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(2, 2, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 50, color = 'black'), axis.title = element_text(size = 55), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 

p_soc_list[[plot_idx]] = p_soc
plot_idx = plot_idx + 1

## Third plot the box plot showing the NSE of BINN predicted SOC and observed SOC accross cross-validation folds
Mean_SOC_NSE = as.numeric(cross_validation_soc_nse[ , 2])
p_box_plot = ggplot() + 
    geom_boxplot(aes(x = 'SOC \n Mean NSE', y = Mean_SOC_NSE), color = '#2166AC', linewidth = 2, outlier.shape = 16, width = 0.3, outlier.size = 5) +
    xlab('') +
    ylab('') +
    ggtitle(' ') +
    theme_minimal(base_family = "Helvetica") +  
    # y axis range starts from 0
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
    # Add NSE on top of the box plot
    annotate("text", x = 1, y = 0.7, label = paste(round(mean(Mean_SOC_NSE), 2)), size = 20, hjust = 0.5, vjust = 0) +
    # coord_flip() +
    theme(plot.background = element_blank()) +
    theme(axis.line = element_line(size = 1, color = 'black')) +
    theme(axis.text = element_text(size = 50, color = 'black')) +
    theme(axis.title = element_text(size = 55, color = 'black')) +
    theme(legend.position = 'None') +
    theme(plot.title = element_text(size = 55, hjust = 0.5)) +
    theme(plot.margin = unit(c(2, 2, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 55, color = 'black'), axis.title = element_text(size = 55), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 


p_soc_list[[plot_idx]] = p_box_plot

# p_soc_combined <- p_soc_list[[1]] / (p_soc_list[[2]] + p_soc_list[[3]])
bottom_row <- plot_grid(p_soc_list[[2]], p_soc_list[[3]], ncol = 2, labels = c('(b)', '(c)'), label_size = 50, label_x = 0.02, label_y = 0.95)
grid_plots <- plot_grid(p_soc_list[[1]], bottom_row, ncol = 1, labels = c('(a)'), label_size = 50, label_x = 0.02, label_y = 0.95, rel_heights = c(1, 1))

# Save the plot
jpeg(paste(cross_validation_dir_output, 'SOC_performance_map_scatter_box_plot.jpg', sep = ''), width = 32, height = 30, units = 'in', res = 300)
print(grid_plots)
dev.off() 


#####################################
# Plot the mean and uncertainty map #
#####################################
# Initialize the plot
p_uncertainty_list = list()
plot_uncertainty_idx = 1

## First plot the map of the difference between the observed and predicted SOC
## Depth Data for each soc observation
soc_upper_depth_test = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_upper_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_upper_depth_test = data.matrix(soc_upper_depth_test)
soc_upper_depth_val = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Validation/nn_val_upper_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_upper_depth_val = data.matrix(soc_upper_depth_val)
soc_upper_depth_train = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Train/nn_train_upper_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_upper_depth_train = data.matrix(soc_upper_depth_train)
soc_upper_depth = soc_upper_depth_train
nan_loc = which(is.na(soc_upper_depth) == 1)
soc_upper_depth[nan_loc] = soc_upper_depth_val[nan_loc]
nan_loc = which(is.na(soc_upper_depth) == 1)
soc_upper_depth[nan_loc] = soc_upper_depth_test[nan_loc]

soc_lower_depth_test = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_lower_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_lower_depth_test = data.matrix(soc_lower_depth_test)
soc_lower_depth_val = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Validation/nn_val_lower_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_lower_depth_val = data.matrix(soc_lower_depth_val)
soc_lower_depth_train = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Train/nn_train_lower_depth_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
soc_lower_depth_train = data.matrix(soc_lower_depth_train)
soc_lower_depth = soc_lower_depth_train
nan_loc = which(is.na(soc_lower_depth) == 1)
soc_lower_depth[nan_loc] = soc_lower_depth_val[nan_loc]
nan_loc = which(is.na(soc_lower_depth) == 1)
soc_lower_depth[nan_loc] = soc_lower_depth_test[nan_loc]

binn_lon_test = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_lons_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lon_test = data.matrix(binn_lon_test)
binn_lon_val = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Validation/nn_val_lons_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lon_val = data.matrix(binn_lon_val)
binn_lon_train = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Train/nn_train_lons_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lon_train = data.matrix(binn_lon_train)
binn_lon = binn_lon_train
nan_loc = which(is.na(binn_lon) == 1)
binn_lon[nan_loc] = binn_lon_val[nan_loc]
nan_loc = which(is.na(binn_lon) == 1)
binn_lon[nan_loc] = binn_lon_test[nan_loc]

binn_lat_test = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Test/nn_test_lats_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lat_test = data.matrix(binn_lat_test)
binn_lat_val = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Validation/nn_val_lats_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lat_val = data.matrix(binn_lat_val)
binn_lat_train = read.csv(paste(cross_validation_dir_input, selected_cross_validation_case, '/Train/nn_train_lats_', selected_cross_validation_case, '.csv', sep = ''), header = FALSE, sep = ',')
binn_lat_train = data.matrix(binn_lat_train)
binn_lat = binn_lat_train
nan_loc = which(is.na(binn_lat) == 1)
binn_lat[nan_loc] = binn_lat_val[nan_loc]
nan_loc = which(is.na(binn_lat) == 1)
binn_lat[nan_loc] = binn_lat_test[nan_loc]

# Select rows where there's at least one valid soc prediction (among 200 columns)
valid_profile_row = which(rowSums(!is.na(binn_lon)) > 0)
# Stop if the number of valid rows is not equal to the max_profile_num
if (length(valid_profile_row) != max_profile_num) {
  stop('Number of valid profiles is not equal to max_profile_num')
}

soc_upper_depth = soc_upper_depth[valid_profile_row, ]
soc_lower_depth = soc_lower_depth[valid_profile_row, ]
binn_lon = binn_lon[valid_profile_row, ]
binn_lat = binn_lat[valid_profile_row, ]

# For each row, if valid_profile_loc is valid in that row, calculate the difference between the observation and prediction and times the difference between the upper and lower depth
# If the valid_profile_loc is not valid in that row, set the difference to be NA
# Store the sum of the difference in a new vector
soc_diff = array(NA, dim = c(max_profile_num, length(cross_validation_dir_list)))

for (i in 1:length(cross_validation_dir_list)){
  for (m in 1:max_profile_num) {
    temp_soc_diff_sum = 0
    for (n in 1:max_sample_num) {
      if (!is.na(cross_validation_binn_soc[i, m, n])) {
        temp_soc_diff_sum = temp_soc_diff_sum + (cross_validation_obs_soc[m, n] - cross_validation_binn_soc[i, m, n]) * (soc_lower_depth[m, n] - soc_upper_depth[m, n])
      }
    }
    soc_diff[m, i] = temp_soc_diff_sum
  }
}

# Calculate the mean and standard deviation of the SOC predictions across cross-validation folds
mean_soc_diff = rep(NA, max_profile_num)
sd_soc_diff = rep(NA, max_profile_num)

for (i in 1:max_profile_num) {
    mean_soc_diff[i] = mean(soc_diff[i, ], na.rm = TRUE)
    sd_soc_diff[i] = sd(soc_diff[i, ], na.rm = TRUE)
}

# Calculate the uncertainty as the coefficient of variation (CV)
cv_soc_diff = sd_soc_diff / abs(mean_soc_diff)

# Get the lon and lat for plotting
current_data_binn = cbind(binn_lon, binn_lat)
colnames(current_data_binn) = c('lon', 'lat')

# Bind the lon and lat with the soc_diff
current_diff_mean = cbind(current_data_binn, mean_soc_diff)
colnames(current_diff_mean) = c('lon', 'lat', 'mean_soc_diff')
current_diff_cv = cbind(current_data_binn, cv_soc_diff)
colnames(current_diff_cv) = c('lon', 'lat', 'cv_soc_diff')

# Convert to data frame
current_diff_mean = data.frame(current_diff_mean)
current_diff_cv = data.frame(current_diff_cv)


#-------------------------------------soc stock and Residence Time map
world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/cb_2018_us_state_500k/cb_2018_us_state_500k.shp', layer = 'cb_2018_us_state_500k')
# world_coastline <- us_map(regions = "states")
# Define the bounding box for the mainland U.S. (excluding Alaska)
# coord_info = '+proj=robin'
# Albers projection
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
lat_limits = rbind(c(-75, 21), c(-130, 48))
# if using Albers projection
# lat_limits = rbind(c(24.5, -62), c(50, -140))
# lat_limits = rbind(c(0, -56), c(0, 80))
# lat_limits_robin = project(xy = as.matrix(lat_limits), proj = coord_info) 
lat_limits_albers = project(xy = as.matrix(lat_limits), proj = coord_info)

# transfer lon and lat to robinson projection 
lon_lat_transfer = project(xy = as.matrix(current_diff_mean[ , c('lon', 'lat')]), proj = coord_info) 
current_diff_mean[ , c('lon', 'lat')] = lon_lat_transfer
# plot data only within the shapefile constraint
current_diff_mean_us <- st_as_sf(current_diff_mean, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_diff_mean_us <- st_intersection(current_diff_mean_us, world_coastline)

# For cv
lon_lat_transfer = project(xy = as.matrix(current_diff_cv[ , c('lon', 'lat')]), proj = coord_info) 
current_diff_cv[ , c('lon', 'lat')] = lon_lat_transfer
current_diff_cv_us <- st_as_sf(current_diff_cv, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_diff_cv_us <- st_intersection(current_diff_cv_us, world_coastline)

# Extract the coordinates from the geometry column
coords <- st_coordinates(current_diff_mean_us$geometry)
# Add lon and lat back to the data frame to the first two columns
current_diff_mean_us$lat <- coords[, 2]
current_diff_mean_us$lon <- coords[, 1]
# move the longtitude to the first column
current_diff_mean_us <- current_diff_mean_us[c("lon", "lat", setdiff(names(current_diff_mean_us), c("lon", "lat")))]
# remove the geometry column
current_diff_mean_us <- st_drop_geometry(current_diff_mean_us)
# remove all column after the 3rd column
current_diff_mean_us <- current_diff_mean_us[ , 1:3]

# For cv
coords <- st_coordinates(current_diff_cv_us$geometry)
current_diff_cv_us$lat <- coords[, 2]
current_diff_cv_us$lon <- coords[, 1]
current_diff_cv_us <- current_diff_cv_us[c("lon", "lat", setdiff(names(current_diff_cv_us), c("lon", "lat")))]
current_diff_cv_us <- st_drop_geometry(current_diff_cv_us)
current_diff_cv_us <- current_diff_cv_us[ , 1:3]

# Normalized the difference between -1 and 1
max_soc = max(quantile(current_diff_mean_us$mean_soc_diff, 0.99, na.rm = TRUE), na.rm = TRUE)
min_soc = min(quantile(current_diff_mean_us$mean_soc_diff, 0.01, na.rm = TRUE), na.rm = TRUE)
# For positive difference, normalized to 0 to 1
for (i in 1:nrow(current_diff_mean_us)) {
	if (current_diff_mean_us$mean_soc_diff[i] > 0) {
		current_diff_mean_us$normalized_soc_diff[i] = current_diff_mean_us$mean_soc_diff[i] / max_soc
	} else {
		current_diff_mean_us$normalized_soc_diff[i] = -1 * current_diff_mean_us$mean_soc_diff[i] / min_soc
	}
}

# Set all difference to -1 if the difference is less than -1, all difference greater than 1 to 1
current_diff_mean_us$normalized_soc_diff[current_diff_mean_us$normalized_soc_diff < -1] = -1
current_diff_mean_us$normalized_soc_diff[current_diff_mean_us$normalized_soc_diff > 1] = 1

# Set the lower and upper limit for the legend
legend_lower_diff_soc = min(current_diff_mean_us$normalized_soc_diff, na.rm = TRUE)
legend_upper_diff_soc = max(current_diff_mean_us$normalized_soc_diff, na.rm = TRUE)

# Plot the difference map for SOC
map_diff_soc = ggplot() +
	# geom_tile(data = current_data_binn_us, aes(x = lon, y = lat, fill = normalized_soc_diff), height = 60000, width = 60000, na.rm = TRUE) +
	geom_point(data = current_diff_mean_us, aes(x = lon, y = lat, color = normalized_soc_diff), size = 3, alpha = 0.8) +
	# scale_fill_gradientn(name = 'SOC Difference', colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
	# Use diff.colors for the colorbar
	scale_color_gradientn(name = 'SOC Difference', colours = diff.colors(15), na.value="transparent", limits = c(legend_lower_diff_soc, legend_upper_diff_soc), trans = 'identity', oob = scales::squish) +
	geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 0.6) + 
	# geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
	# coord_sf(xlim = lat_limits_robin[ , 1], ylim = lat_limits_robin[ , 2], datum = NA) +
  # if using Albers projection
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
	# change the background to black and white
	# coord_equal() +
	# theme_map() +
	# ylim(lat_limits_albers[ , 2]) +
	# change the legend properties
	# theme(legend.position = 'none') +
	theme(legend.justification = c(0, 0), legend.position = c(-0.1, 0.5), legend.background = element_rect(fill = NA), legend.text.align = 0, legend.key.height = unit(1.2, 'cm'), legend.key.width = unit(1, 'cm')) +
	# theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
	# change the size of colorbar
	guides(color = guide_colorbar(direction = 'vertical', barwidth = 4, barheight = 16, title = 'SOC\nDifference', title.position = 'top', title.hjust = 0, title.vjust = 2, frame.linewidth = 0), reverse = FALSE) +
  theme(legend.text = element_text(size = 50, ), legend.title = element_text(size = 0)) +
	# add title
	labs(title = 'SOC Difference Map') +
	# modify the position of title
	theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 55)) + 
	# modify the font size
	# theme(axis.title = element_text(size = 30)) + 
	theme(axis.title = element_blank()) +
	theme(panel.background = element_rect(fill = NA, colour = NA)) +
	# modify the margin
	theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
	theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))
	# theme(axis.text=element_text(size = 15, color = 'black'))

p_uncertainty_list[[plot_uncertainty_idx]] = map_diff_soc
plot_uncertainty_idx = plot_uncertainty_idx + 1

## Second plot the difference map for the cv
# Check the distribution of cv_soc_diff$cv_soc_diff
cap_values = 10
map_diff_cv_soc = ggplot() +
  geom_point(
    data = current_diff_cv_us,
    aes(x = lon, y = lat, color = pmin(cv_soc_diff, cap_values)),  # cap values >1000
    size = 3,
    alpha = 0.8
  ) +
	scale_color_gradientn(
  name = 'SOC Difference',
  colours = diff.colors(15)[8:15],
  limits = c(0, cap_values),
  oob = scales::squish,
  breaks = c(0, 2.5, 5, 7.5, cap_values),
  labels = c("0", "2.5", "5", "7.5", paste0("≥", cap_values))
  ) +
	geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 0.6) + 
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
	theme(legend.justification = c(0, 0), legend.position = c(-0.1, 0.5), legend.background = element_rect(fill = NA), legend.text.align = 0, legend.key.height = unit(1.2, 'cm'), legend.key.width = unit(1, 'cm')) +
	guides(color = guide_colorbar(direction = 'vertical', barwidth = 4, barheight = 16, title = 'SOC\nDifference', title.position = 'top', title.hjust = 0, title.vjust = 2, frame.linewidth = 0), reverse = FALSE) +
  theme(legend.text = element_text(size = 50, ), legend.title = element_text(size = 0)) +
	# add title
	labs(title = 'Uncertainty Map (Coefficient of Variation)') +
	# modify the position of title
	theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 55)) + 
	# modify the font size
	# theme(axis.title = element_text(size = 30)) + 
	theme(axis.title = element_blank()) +
	theme(panel.background = element_rect(fill = NA, colour = NA)) +
	# modify the margin
	theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
	theme(plot.margin = unit(c(0, 0, 0, 0), 'inch'))

p_uncertainty_list[[plot_uncertainty_idx]] = map_diff_cv_soc
plot_uncertainty_idx = plot_uncertainty_idx + 1

## Third plot the scatter plot of the observed and predicted SOC
# Calculate the SOC NSE
current_data = cbind(as.vector(mean_cross_validation_binn_soc), as.vector(cross_validation_obs_soc))/1000
current_data = data.frame(current_data[complete.cases(current_data), ])
colnames(current_data) = c('binn', 'obs')
# Calculate the SOC NSE
nse_process_middle = 1 - sum((current_data$binn - current_data$obs)^2)/sum((mean(current_data$obs) - current_data$obs)^2)
# If plot within 5–95th percentile on both obs and binn
qs_x <- quantile(current_data$obs,  c(.05, .95), na.rm = TRUE)
qs_y <- quantile(current_data$binn, c(.05, .95), na.rm = TRUE)

current_data_inner <- current_data %>%
  mutate(inlier = dplyr::between(obs,  qs_x[1], qs_x[2]) &
                  dplyr::between(binn, qs_y[1], qs_y[2]))

# Define the binwidth for x and y axis
x_rng <- range(current_data$obs,  na.rm = TRUE)
y_rng <- range(current_data$binn, na.rm = TRUE)
binwidth_xy <- c(diff(x_rng) / 100, diff(y_rng) / 100)

p_soc = ggplot(data = current_data_inner) + 
    stat_bin_hex(aes(x = obs, y = binn, fill = after_stat(count)), bins = 100) +
    scale_fill_gradientn(name = 'Count', colors = viridis(7), trans = 'identity', limits = c(1, 20), oob = scales::squish) +
    scale_y_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
    scale_x_continuous(limits = c(0.1, 1000), trans = 'log10', labels = trans_format('log10', math_format(10^.x))) + 
    # # Outliers: gray points (no legend)
    # stat_bin_hex(
    #   data = filter(current_data_inner, !inlier),
    #   aes(x = obs, y = binn),
    #   # binwidth = binwidth_xy,
    #   bins = 100,
    #   fill = "grey75", color = NA, alpha = 0.7,  
    #   show.legend = FALSE
    # ) +
    geom_abline(slope = 1, intercept = 0, size = 1, color = 'black') +
    theme_classic() + 
    # add title
    labs(title = paste('BINN vs Observed SOC (NSE: ', round(nse_process_middle, 2), ')', sep = ''),
    x = expression(paste('Observed SOC (kg C m'^'-3', ')', sep = '')), y = expression(paste('BINN simulation (kg C m'^'-3', ')', sep = ''))) +
    # change the legend properties
    guides(fill = guide_colorbar(direction = 'horizontal', barwidth = 15, barheight = 2.5, title.position = 'right', title.hjust = 0, title.vjust = 0.8, label.hjust = 0.5, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 35), legend.title = element_text(size = 35))  +
    theme(legend.justification = c(0, 0), legend.position = c(0, 0.9), legend.background = element_rect(fill = NA)) + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, size = 55)) + 
    # modify the font size
    # modify the margin
    # theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), 'inch')) +
    theme(axis.text=element_text(size = 50, color = 'black'), axis.title = element_text(size = 55), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 

p_uncertainty_list[[plot_uncertainty_idx]] = p_soc
plot_uncertainty_idx = plot_uncertainty_idx + 1

## Third plot the box plot showing the NSE of BINN predicted SOC and observed SOC accross cross-validation folds
Mean_SOC_NSE = as.numeric(cross_validation_soc_nse[ , 2])
p_box_plot = ggplot() + 
    geom_boxplot(aes(x = 'SOC \n Mean NSE', y = Mean_SOC_NSE), color = '#2166AC', linewidth = 2, outlier.shape = 16, width = 0.3, outlier.size = 5) +
    xlab('') +
    ylab('') +
    ggtitle('Mean Test Performance') +
    theme_minimal(base_family = "Helvetica") +  
    # y axis range starts from 0
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
    # Add NSE on top of the box plot
    annotate("text", x = 1, y = 0.75, label = paste(round(mean(Mean_SOC_NSE), 2)), size = 18, hjust = 0.5, vjust = 0) +
    # coord_flip() +
    theme(plot.background = element_blank()) +
    theme(axis.line = element_line(size = 1, color = 'black')) +
    theme(axis.text = element_text(size = 50, color = 'black')) +
    theme(axis.title = element_text(size = 55, color = 'black')) +
    theme(legend.position = 'None') +
    theme(plot.title = element_text(size = 55, hjust = 0.5)) +
    theme(plot.margin = unit(c(0, 0, 0.2, 0), 'inch')) +
    theme(axis.text=element_text(size = 55, color = 'black'), axis.title = element_text(size = 55), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 


p_uncertainty_list[[plot_uncertainty_idx]] = p_box_plot

# p_soc_combined <- p_soc_list[[1]] / (p_soc_list[[2]] + p_soc_list[[3]])
bottom_row <- plot_grid(p_uncertainty_list[[3]], p_uncertainty_list[[4]], ncol = 2, labels = c('(c)', '(d)'), label_size = 60, label_x = 0, label_y = 1.05)
top_row <- plot_grid(p_uncertainty_list[[1]], p_uncertainty_list[[2]], ncol = 1, labels = c('(a)', '(b)'), label_size = 60, label_x = 0, label_y = 0.95)
grid_plots <- plot_grid(top_row, bottom_row, ncol = 1, labels = c(''), label_size = 70, label_x = 0, label_y = 0.1, rel_heights = c(7, 3))

# Save the plot
jpeg(paste(cross_validation_dir_output, 'SOC_Uncertainty_map_scatter_box_plot.jpg', sep = ''), width = 32, height = 50, units = 'in', res = 300)
print(grid_plots)
dev.off() 

# Report the mean and standard deviation of the cross_validation_soc_nse
mean_nse = mean(Mean_SOC_NSE, na.rm = TRUE)
sd_nse = sd(Mean_SOC_NSE, na.rm = TRUE)
print(paste('Mean NSE:', round(mean_nse, 3)))
print(paste('Standard Deviation of NSE:', round(sd_nse, 3)))