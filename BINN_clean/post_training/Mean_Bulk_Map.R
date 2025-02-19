# Average accross all the cross validation results and plot the mean bulk map
# Compare BINN results with PRODA results

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

rm(list = ls())

setwd('D:/Research/BINN/BINN_output/plot/')
Sys.setenv(PROJ_LIB = "C:/Users/hx293/AppData/Local/R/win-library/4.3/sf/proj")

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
cross_validation_folder = 'Cross_Validation_seed_111'
cross_validation_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/')
cross_validation_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/Output/')
# Get the list of all the folders in the cross validation directory
cross_validation_dir_list = list.dirs(cross_validation_dir_input, full.names = FALSE, recursive = FALSE)
# Exclude the folder of Output
cross_validation_dir_list = cross_validation_dir_list[!cross_validation_dir_list %in% 'Output']


# PRODA data path
data_dir_PRODA = 'D:/Nutstore/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
data_dir_loc = 'D:/Nutstore/Research_Data/BINN/Server_Script/post_training/component_calculation/'
# data_dir_PRODA = 'C:/Research_Data/BINN/Server_Script/post_training/soc_component_proda/soc_component_proda/'
# data_dir_loc = 'C:/Research_Data/BINN/Server_Script/post_training/component_calculation/'
# create output folder if not exist
if (!dir.exists(cross_validation_dir_output)) {
  dir.create(cross_validation_dir_output)
}

#################################################################################
# Bulk Processes
#################################################################################

## BINN ##
# Import lon and lat from one of the cross validation
binn_lon = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[1], '/Prediction', '/nn_grid_', 'lons_', 
                            cross_validation_dir_list[1], '.csv', sep = ''), header = FALSE)
binn_lat = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[1], '/Prediction', '/nn_grid_', 'lats_',
                            cross_validation_dir_list[1], '.csv', sep = ''), header = FALSE)
current_data_binn = cbind(binn_lon, binn_lat)
colnames(current_data_binn) = c('lon', 'lat')

# load BINN data: bulk_A, bulk_I, bulk_K, bulk_V, bulk_xi, carbon_input, litter_fraction
input_matrix = c('bulk_A', 'bulk_I', 'bulk_K', 'bulk_V', 'bulk_xi', 'carbon_input', 'litter_fraction')

# Average accross all the cross validation results for Bulk Processes
bulk_process_binn = array(NA, dim = c(nrow(binn_lon), length(input_matrix), length(cross_validation_dir_list)))
icross_valid = 1
for (icross_valid in 1:length(cross_validation_dir_list)) {
  for (i in 1:length(input_matrix)) {
    # For Bulk I, we choose to import the parameter values from the 21st column of csv file: nn_grid_pred_para
    if (input_matrix[i] == 'bulk_I') {
      temp_bulk = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[icross_valid], '/Prediction', '/nn_grid_pred_para_',
                                 cross_validation_dir_list[icross_valid], '.csv', sep = ''), header = FALSE)
      temp_bulk = as.data.frame(temp_bulk[ , 21])
      bulk_process_binn[ , i, icross_valid] = temp_bulk[ , 1]
    } else {
      temp_bulk = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[icross_valid], '/Prediction', '/nn_grid_bulk_', input_matrix[i], '_',
                                 cross_validation_dir_list[icross_valid], '.csv', sep = ''), header = FALSE)
      bulk_process_binn[ , i, icross_valid] = temp_bulk[ , 1]
    }
  }
}

# Calculate the mean of the bulk processes
bulk_process_mean_binn = apply(bulk_process_binn, c(1, 2), median, na.rm = TRUE)
current_data_binn = cbind(current_data_binn, bulk_process_mean_binn)
dim(bulk_process_mean_binn)
dim(current_data_binn)
# Remove the NA values
current_data_binn = current_data_binn[complete.cases(current_data_binn), ]
dim(current_data_binn)


## PRODA ##
model_name = 'cesm2_clm5_cen_vr_v2'
nn_exp_name = 'exp_pc_cesm2_23'
time_domain = 'whole_time'

valid_grid_loc_clm = read.csv(paste(data_dir_loc, 'neural_networking/valid_grid_loc_', model_name, '_', time_domain, '_', nn_exp_name, '_cross_valid_0_', as.character(1), '.csv', sep = ''), header = FALSE)
valid_grid_loc_clm = valid_grid_loc_clm$V1

global_lat_lon_clm = readMat(paste(data_dir_PRODA, 'soc_simu_grid_info_', model_name, '_', nn_exp_name, '_cross_valid_0_', as.character(1), '.mat', sep = ''))
global_lat_lon_clm = global_lat_lon_clm$var.data.middle[ , 1:2]
colnames(global_lat_lon_clm) = c('lon', 'lat')

bulk_process_clm = array(NA, dim = c(nrow(global_lat_lon_clm), 7, 10))

icross_valid = 2
for (icross_valid in 1:10) {
  global_simu = readMat(paste(data_dir_PRODA, 'bulk_process_summary_', model_name, '_', nn_exp_name, '_cross_valid_0_', as.character(icross_valid), '.mat', sep = ''))
  bulk_process_clm[ , , icross_valid] = global_simu$var.data.middle[ , c(1:7)]
}
bulk_process_mean_clm = apply(bulk_process_clm, c(1, 2), median, na.rm = TRUE)

# print the head of the data
head(bulk_process_mean_clm)
dim(bulk_process_mean_clm)

# Change the bulk I (the 2nd column) to the 21st column of the average among the 10 cross validation
icross_para = 1
grid_PRODA_para = array(NA, dim = c(nrow(global_lat_lon_clm), 10)) 
for (icross_para in 1:10) {
  temp_para = read.csv(paste(data_dir_loc, '/neural_networking/grid_para_result_', model_name, '_', time_domain, '_', nn_exp_name, '_cross_valid_0_', as.character(icross_para), '.csv', sep = ''), header = FALSE)
  temp_para = as.data.frame(temp_para[ , 21])
  grid_PRODA_para[ , icross_para] = temp_para[ , 1]
}

# Check the dimension of the data
head(grid_PRODA_para)
dim(grid_PRODA_para)

# Switch the second column of the bulk_process_mean_clm to the average of each row of grid_PRODA_para
bulk_process_mean_clm[ , 2] = apply(grid_PRODA_para, 1, mean, na.rm = TRUE)

# Check the dimension of the data
head(bulk_process_mean_clm)
dim(bulk_process_mean_clm)


#################################################################################
# plot figures
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

# transfer lon and lat to robinson projection 

## Proda
# process map clm
current_data_clm = data.frame(cbind(global_lat_lon_clm, bulk_process_mean_clm))
colnames(current_data_clm) = c('lon', 'lat', 'A', 'I', 'K', 'V', 'Xi', 'NPP', 'F')
lon_lat_transfer = project(xy = as.matrix(current_data_clm[ , c('lon', 'lat')]), proj = coord_info) 
current_data_clm[ , c('lon', 'lat')] = lon_lat_transfer
# plot data only within the shapefile constraint
current_data_clm_us <- st_as_sf(current_data_clm, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_data_clm_us <- st_intersection(current_data_clm_us, world_coastline)
# current_data_clm_us <- st_join(current_data_clm_us, world_coastline, join = st_within_distance, dist = 1000)
# Extract the coordinates from the geometry column
coords <- st_coordinates(current_data_clm_us$geometry)
# Add lon and lat back to the data frame to the first two columns
current_data_clm_us$lat <- coords[, 2]
current_data_clm_us$lon <- coords[, 1]
# move the longtitude to the first column
current_data_clm_us <- current_data_clm_us[c("lon", "lat", setdiff(names(current_data_clm_us), c("lon", "lat")))]
# remove the geometry column
current_data_clm_us <- st_drop_geometry(current_data_clm_us)
# remove all column after the 9th column
current_data_clm_us <- current_data_clm_us[ , 1:9]
head(current_data_clm_us)
tail(current_data_clm_us)



## BINN
lon_lat_transfer = project(xy = as.matrix(current_data_binn[ , c('lon', 'lat')]), proj = coord_info) 
current_data_binn[ , c('lon', 'lat')] = lon_lat_transfer
# plot data only within the shapefile constraint
current_data_binn_us <- st_as_sf(current_data_binn, coords = c('lon', 'lat'), crs = st_crs(world_coastline))
current_data_binn_us <- st_intersection(current_data_binn_us, world_coastline)
# current_data_binn_us <- st_join(current_data_binn_us, world_coastline, join = st_within_distance, dist = 1000)
# Extract the coordinates from the geometry column
coords <- st_coordinates(current_data_binn_us$geometry)
# Add lon and lat back to the data frame to the first two columns
current_data_binn_us$lat <- coords[, 2]
current_data_binn_us$lon <- coords[, 1]
# move the longtitude to the first column
current_data_binn_us <- current_data_binn_us[c("lon", "lat", setdiff(names(current_data_binn_us), c("lon", "lat")))]
# remove the geometry column
current_data_binn_us <- st_drop_geometry(current_data_binn_us)
# remove all column after the 9th column
current_data_binn_us <- current_data_binn_us[ , 1:9]

# Select the interested rows in both data based on the lon and lat

# Check dimension
dim(current_data_clm_us)
dim(current_data_binn_us)

# Combine lon and lat for the two data
current_data_clm_us[ , c('lon_lat')] = paste(current_data_clm_us$lon, current_data_clm_us$lat, sep = '_')
current_data_binn_us[ , c('lon_lat')] = paste(current_data_binn_us$lon, current_data_binn_us$lat, sep = '_')
# Select the interested rows in both data based on the lon and lat
interested_lon_lat = intersect(current_data_clm_us$lon_lat, current_data_binn_us$lon_lat)
current_data_clm_us = current_data_clm_us[current_data_clm_us$lon_lat %in% interested_lon_lat, ]
current_data_binn_us = current_data_binn_us[current_data_binn_us$lon_lat %in% interested_lon_lat, ]
# Check dimension
dim(current_data_clm_us)
dim(current_data_binn_us)


####################################################################

process_scale_option = c('identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity', 'identity')

process_name =  c('Carbon Transfer Efficiency', 
                  'Carbon Input Allocation', 
                  'Baseline Decomposition', 
                  'Vertical Transport Rate',
                  'Environmental Modifier', 
                  'Plant Carbon Inputs', 
                  'Litter to Mineral Soil Fraction')
process_unit = c('unitless',
                 'unitless', 
                 expression(paste('yr'^'-1', sep = '')),
                 expression(paste('yr'^'-1', sep = '')),
                 'unitless', 
                 expression(paste('gCm'^'-2', ' yr'^'-1', sep = '')), 
                 'unitless')

corr_process_summary = array(NA, dim = c(length(process_name), 2))

ipara = 3

# Begin Plotting
for (ipara in 1:length(process_name)){

  # BINN
  middle_data_binn = current_data_binn_us[ , c(1:2, 2 + ipara)]
  colnames(middle_data_binn) = c('lon', 'lat', 'process')
  
  legend_lower_binn = apply(current_data_binn_us[ , c(3:9)], 2, quantile, prob = 0.05, na.rm = TRUE)
  legend_upper_binn = apply(current_data_binn_us[ , c(3:9)], 2, quantile, prob = 0.95, na.rm = TRUE)


  # PRODA
  middle_data_clm = current_data_clm_us[ , c(1, 2, (ipara+2))]
  colnames(middle_data_clm) = c('lon', 'lat', 'process')
  
  legend_lower_clm = apply(current_data_clm_us[ , c(3:9)], 2, quantile, prob = 0.05, na.rm = TRUE)
  legend_upper_clm = apply(current_data_clm_us[ , c(3:9)], 2, quantile, prob = 0.95, na.rm = TRUE)
  
  legend_lower_clm = apply(rbind(legend_lower_clm, legend_lower_binn), 2, min)
  legend_upper_clm = apply(rbind(legend_upper_clm, legend_upper_binn), 2, max)
  
  legend_lower_binn = legend_lower_clm
  legend_upper_binn = legend_upper_clm
  

  # plot figure
  p_binn =
    ggplot() +
    geom_tile(data = middle_data_binn, aes(x = lon, y = lat, fill = process), height = 60000, width = 60000, na.rm = TRUE) +
    scale_fill_gradientn(name = process_unit[ipara], colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_binn[ipara], legend_upper_binn[ipara]), trans = process_scale_option[ipara], oob = scales::squish) +
    geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
	  # if only plot the mainland US
	  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
    # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
    # change the background to black and white
    # coord_equal() +
    # theme_map() +
    ylim(lat_limits_albers[ , 2]) +
    # change the legend properties
    # theme(legend.position = 'none') +
    theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
    # theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
    # change the size of colorbar
    guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
    # add title
    labs(title = paste('BINN: ', process_name[ipara], sep = ''), x = '', y = '') + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
    # modify the font size
    theme(axis.title = element_text(size = 20)) + 
    theme(panel.background = element_rect(fill = NA, colour = NA)) +
    # modify the margin
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0, 0, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 35, color = 'black'))
  
  # PRODA
  p_clm =
    ggplot() +
    geom_tile(data = middle_data_clm, aes(x = lon, y = lat, fill = process), height = 60000, width = 60000, na.rm = TRUE) +
    scale_fill_gradientn(name = process_unit[ipara], colours = rev(viridis(15)), na.value="transparent", limits = c(legend_lower_clm[ipara], legend_upper_clm[ipara]), trans = process_scale_option[ipara], oob = scales::squish) +
    geom_sf(data = world_coastline, fill = NA, color = 'black', linewidth = 1) + 
    # geom_polygon(data = world_ocean, aes(x = lon, y = lat), fill = NA, color = 'black', size = 2) +
	  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
    # change the background to black and white
    # coord_equal() +
    # theme_map() +
    ylim(lat_limits_albers[ , 2]) +
    # change the legend properties
    # theme(legend.position = 'none') +
    theme(legend.justification = c(0, 0), legend.position = c(-0.03, 0.02), legend.background = element_rect(fill = NA), legend.text.align = 0) +
    # theme(legend.justification = c(0.5, 0), legend.position = c(0.5, 0), legend.background = element_rect(fill = NA), legend.direction = 'horizontal') +
    # change the size of colorbar
    guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
    theme(legend.text = element_text(size = 30, ), legend.title = element_text(size = 35)) +
    # add title
    labs(title = paste('PRODA: ', process_name[ipara], sep = ''), x = '', y = '') + 
    # modify the position of title
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 40)) + 
    # modify the font size
    theme(axis.title = element_text(size = 20)) + 
    theme(panel.background = element_rect(fill = NA, colour = NA)) +
    # modify the margin
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    theme(plot.margin = unit(c(0, 0, 0, 0), 'inch')) +
    theme(axis.text=element_text(size = 35, color = 'black'))

	# correlation
	middle_data_corr = cbind(middle_data_clm$process, middle_data_binn$process)
	middle_data_corr = data.frame(middle_data_corr)
	colnames(middle_data_corr) = c('clm', 'binn')
  # Calculate the correlation
  corr_process_middle = cor.test(middle_data_corr$clm, middle_data_corr$binn, na.rm = TRUE)
	
	if (ipara == 3) {
		limit_clm = c(0.001, 1)
		limit_binn = c(0.001, 1)
	} else {
		limit_clm = quantile(middle_data_clm$process, probs = c(0, 1), na.rm = TRUE)
		limit_binn = quantile(middle_data_binn$process, probs = c(0, 1), na.rm = TRUE)
	}
	limit_clm = c(min(limit_clm[1], limit_binn[1]), max(limit_clm[2], limit_binn[2]))
	limit_binn = limit_clm
	
	corr_process_middle = cor.test(middle_data_corr$clm, middle_data_corr$binn, na.rm = TRUE)
	corr_process_summary[ipara, ] = c(corr_process_middle$estimate, corr_process_middle$p.value)
	p_corr =
		ggplot() + 
		stat_bin_hex(data = middle_data_corr, aes(x = clm, y = binn), bins = 100) +
		scale_fill_gradientn(name = 'Count', colors = viridis(7), trans = 'identity', oob = scales::squish) +
		geom_abline(slope = 1, intercept = 0, size = 2, color = 'black') +
		scale_x_continuous(trans = process_scale_option[ipara], limits = limit_clm) +
		scale_y_continuous(trans = process_scale_option[ipara], limits = limit_binn) +
		theme_classic() + 
		# add title
		labs(title = '', x = 'PRODA', y = 'BINN') + 
    # add correlation information
    annotate('text', label = paste('Correlation: ', round(corr_process_middle$estimate, 2), sep = ''), size = 12, x = -Inf, y = Inf, hjust = 0, vjust = 1) +
		# change the legend properties
		guides(fill = guide_colorbar(direction = 'horizontal', barwidth = 15, barheight = 2.5, title.position = 'right', title.hjust = 0, title.vjust = 0.8, label.hjust = 0.5, frame.linewidth = 0), reverse = FALSE) +
		theme(legend.text = element_text(size = 25), legend.title = element_text(size = 25))  +
		theme(legend.justification = c(1, 0), legend.position = 'None', legend.background = element_rect(fill = NA)) + 
		# modify the position of title
		theme(plot.title = element_text(hjust = 0.5, size = 50)) + 
		# modify the font size
		# modify the margin
		# theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
		theme(plot.margin = unit(c(0., 0.2, 0.2, 0.2), 'inch')) +
		theme(axis.text=element_text(size = 30, color = 'black'), axis.title = element_text(size = 35), axis.line = element_line(size = 1), axis.ticks = element_line(size = 1, color = 'black'), axis.ticks.length = unit(0.12, 'inch')) 
	

  eval(parse(text = paste('p_binn', ipara, ' = p_binn', sep = '')))
  eval(parse(text = paste('p_clm', ipara, ' = p_clm', sep = '')))
  eval(parse(text = paste('p_corr', ipara, ' = p_corr', sep = '')))
  
}


jpeg(paste(cross_validation_dir_output, 'Bulk_Process.jpeg', sep = ''), width = 36, height = 35, units = 'in', res = 300)
plot_grid(p_binn1, p_clm1, p_corr1, NULL, 
          p_binn3, p_clm3, p_corr3, NULL, 
          p_binn5, p_clm5, p_corr5, NULL, 
          p_binn2, p_clm2, p_corr2, NULL, 
          p_binn4, p_clm4, p_corr4, NULL, 
          p_binn6, p_clm6, p_corr6, NULL, 
          nrow = 6, ncol = 4 ,
          rel_widths = c(3, 3, 3, 0.10),
          labels = c('a', 'b', 'c', ' ',
                     'd', 'e', 'f', ' ',
                     'g', 'h', 'i', ' ',
                     'j', 'k', 'l', ' ',
                     'm', 'n', 'o', ' ',
                     'p', 'q', 'r', ' '),
          label_size = 70,
          label_x = 0.05, label_y = 1.05,
          label_fontfamily = 'Arial',
          label_fontface = 'bold'
)
dev.off()

jpeg(paste(cross_validation_dir_output, 'Litter_Fraction.jpeg',sep = ''), width = 36, height = 10, units = 'in', res = 300)
plot_grid(p_binn7, p_clm7, p_corr7, NULL, 
          nrow = 1, ncol = 4 ,
          rel_widths = c(3, 3, 3, 0.10),
          labels = c('a', 'b', 'c', ' '),
          label_size = 70,
          label_x = 0.05, label_y = 1.05,
          label_fontfamily = 'Arial',
          label_fontface = 'bold'
)
dev.off()