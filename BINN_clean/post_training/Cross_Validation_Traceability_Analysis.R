## Packages
library(R.matlab)
library(ggplot2)
library(cowplot)
library(viridis)
library(scales)
library(ncdf4)
library(jpeg)
library(tidyverse)
library(magick)
library(gridExtra)
library(viridis)
library(sf)
library(sp)
library(GGally)
library(raster)
library(proj4)
library(metR)
library(plotly)
library(dplyr)
library(SpatGRID)

# library(av)
# install.packages('rgdal')
# options(timeout = 999999999)


rm(list = ls())
# setwd('C:/Users/Hardy/Documents/BINN/')
# Sys.setenv(PROJ_LIB = "C:/Users/Hardy/AppData/Local/R/win-library/4.4/sf/proj")
setwd('D:/Research/BINN/BINN_output/plot/')
Sys.setenv(PROJ_LIB = "C:/Program Files/R/R-4.3.3/library/sf/proj")

## Jet colorbar function
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
diff.colors <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "white", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"))

# Define a color palette
biome_colors <- c(
  "EASTERN TEMPERATE FORESTS" = "#0072B2",       # Blue
  "GREAT PLAINS" = "#D55E00",                    # Red
  "MARINE WEST COAST FOREST" = "#009E73",        # Green
  "MEDITERRANEAN CALIFORNIA" = "#E69F00",        # Orange
  "NORTH AMERICAN DESERTS" = "#CC79A7",          # Pink
  "NORTHERN FORESTS" = "#56B4E9",                # Light Blue
  "NORTHWESTERN FORESTED MOUNTAINS" = "#F0E442", # Yellow
  "SOUTHERN SEMIARID HIGHLANDS" = "#999999",     # Gray
  "TEMPERATE SIERRAS" = "#8B4513",               # Brown
  "TROPICAL WET FORESTS" = "#7F7FFF"             # Purple-Blue
)

custom_theme <- theme_classic() +
  theme(
    legend.position = "right",
    legend.title = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 80, face = "bold"),
    axis.title = element_text(size = 70),
    axis.text = element_text(size = 60),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = "black"),
    panel.border = element_rect(color = "black", fill = NA)
  )

# Create a dataframe to store the width between two interfaces in CLM5
dz = c(2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002,
		8.000000000000000E-002, 0.120000000000000, 0.160000000000000,
		0.200000000000000, 0.240000000000000, 0.280000000000000,
		0.320000000000000, 0.360000000000000, 0.400000000000000,
		0.440000000000000, 0.540000000000000, 0.640000000000000,
		0.740000000000000, 0.840000000000000, 0.940000000000000,
		1.04000000000000, 1.14000000000000)


#############################################################################
# Data Path
#############################################################################
cross_validation_folder = 'Cross_Validation_seed_111'
cross_validation_dir_input = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/')
cross_validation_dir_output = paste0('D:/Research/BINN/BINN_output/neural_network/', cross_validation_folder, '/Output/')
# cross_validation_dir_input = paste0('C:/Users/Hardy/Documents/BINN//', cross_validation_folder, '/')
# cross_validation_dir_output = paste0('C:/Users/Hardy/Documents/BINN/', cross_validation_folder, '/Output/')
# Get the list of all the folders in the cross validation directory
cross_validation_dir_list = list.dirs(cross_validation_dir_input, full.names = FALSE, recursive = FALSE)
# Exclude the folder of Output
cross_validation_dir_list = cross_validation_dir_list[!cross_validation_dir_list %in% 'Output']


# create output folder if not exist
if (!dir.exists(cross_validation_dir_output)) {
  dir.create(cross_validation_dir_output)
}

#################################################################################
# Traceable Parts Load
#################################################################################

## BINN ##
# Import lon and lat from one of the cross validation
binn_lon = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[1], '/Prediction', '/nn_grid_', 'lons_', 
                            cross_validation_dir_list[1], '.csv', sep = ''), header = FALSE)
binn_lat = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[1], '/Prediction', '/nn_grid_', 'lats_',
                            cross_validation_dir_list[1], '.csv', sep = ''), header = FALSE)
current_data_binn = cbind(binn_lon, binn_lat)
colnames(current_data_binn) = c('lon', 'lat')

# load traceable parts for BINN
# input_matrix = c('carbon_input', 'total_res_time', 'bulk_xi', 'total_res_time_base')
input_matrix = c('cpools_layer', 'carbon_input', 'bulk_xi')
traceable_parts = c('soc_storage', 'carbon_input', 'bulk_xi', 'total_res_time', 'total_res_time_base')

# Average accross all the cross validation results for Bulk Processes
bulk_process_binn = array(NA, dim = c(nrow(binn_lon), length(input_matrix), length(cross_validation_dir_list)))
icross_valid = 1
i = 1
for (icross_valid in 1:length(cross_validation_dir_list)) {
  for (i in 1:length(input_matrix)) {
    temp_bulk = read.csv(paste(cross_validation_dir_input, cross_validation_dir_list[icross_valid], '/Prediction', '/nn_grid_bulk_', input_matrix[i], '_',
                                cross_validation_dir_list[icross_valid], '.csv', sep = ''), header = FALSE)
    # For cpools_layer, soc from each layer will times the thickness of the layer and sum up
    if (input_matrix[i] == 'cpools_layer') {
      for (j in 1:ncol(temp_bulk)) {
        temp_bulk[ , j] = temp_bulk[ , j] * dz[j]
      }
      temp_bulk = rowSums(temp_bulk)
      temp_bulk = as.data.frame(temp_bulk)
    }
    bulk_process_binn[ , i, icross_valid] = temp_bulk[ , 1]
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

# Rename the columns
colnames(current_data_binn)[3:5] = input_matrix

# Calculate the remaining traceable parts
current_data_binn$total_res_time = current_data_binn$cpools_layer / current_data_binn$carbon_input
current_data_binn$total_res_time_base = current_data_binn$total_res_time * current_data_binn$bulk_xi

# Rename the columns
colnames(current_data_binn)[3:7] = traceable_parts
head(current_data_binn, 10)

#################################################################################
# Merge with Biome Data by lon and lat
#################################################################################
# Shapefile for Biomes map
biomes_map = st_read('D:/Nutstore/Research_Data/Map_Plot/na_cec_eco_l1/NA_CEC_Eco_Level1.shp')
# biomes_map = st_read('C:/Research_Data/Map_Plot/na_cec_eco_l1/NA_CEC_Eco_Level1.shp')
# biomes_map = st_read('C:/Research_Data/Map_Plot/global_biomes/tnc_terr_ecoregions.shp')
# Check for invalid geometries
invalid_geometries <- !st_is_valid(biomes_map)
sum(invalid_geometries)
# Make the geometries valid
biomes_map <- st_make_valid(biomes_map)
all(st_is_valid(biomes_map))
# Transform biomes_map to a geographic CRS (WGS84 - EPSG:4326)
biomes_map <- st_transform(biomes_map, crs = 4326)


# Function to create a grid of 0.5 x 0.5-degree resolution based on a lat, lon point
# and calculate the area of each biome within the grid
calculate_biome_area <- function(lat, lon, biome_map) {
  # Create a square grid of 0.5° x 0.5° centered on the lat, lon point
  bbox <- st_bbox(c(
    xmin = lon - 0.25,
    xmax = lon + 0.25,
    ymin = lat - 0.25,
    ymax = lat + 0.25))

  grid <- st_as_sfc(bbox)
  st_crs(grid) <- st_crs(biome_map)

  # Intersect the grid with the biome map
  intersected_biomes <- st_intersection(biome_map, grid)
  
  # Calculate the area of each biome in the grid
  biome_areas <- intersected_biomes %>%
    group_by(NA_L1NAME) %>%  # Assuming the biome type is in a column named 'NAME'
    summarize(area = sum(st_area(.))) %>%
    arrange(desc(area))
  
  return(biome_areas)
}

current_data_binn$biome_type <- NA  # Create a new column for biome type

start_time <- Sys.time()
for (i in 1:nrow(current_data_binn)) {
  lat <- current_data_binn$lat[i]
  lon <- current_data_binn$lon[i]

  biome_areas <- calculate_biome_area(lat, lon, biomes_map)
  
  # Assign the biome type with the largest area
  if (nrow(biome_areas) > 0) {
    current_data_binn$NA_L1NAME[i] <- biome_areas$NA_L1NAME[1]
    
    # Handle case where largest biome type is NA
    if (is.na(current_data_binn$NA_L1NAME[i]) && nrow(biome_areas) > 1) {
      current_data_binn$NA_L1NAME[i] <- biome_areas$NA_L1NAME[2]
    }
  }
  print(paste0("Progress: ", i, "/", nrow(current_data_binn)))
}
end_time <- Sys.time()
print(end_time - start_time)

print(head(current_data_binn, 10))

current_data_binn_backup = current_data_binn

# Save current_data_binn
write.csv(current_data_binn, paste(cross_validation_dir_output, 'current_data_binn.csv', sep = ''), row.names = FALSE)

############################
## Read current_data_binn ##
############################
current_data_binn = read.csv(paste(cross_validation_dir_output, 'current_data_binn.csv', sep = ''), header = TRUE)
print(head(current_data_binn, 10))

world_coastline = st_read('D:/Nutstore/Research_Data/Map_Plot/cb_2018_us_state_500k/cb_2018_us_state_500k.shp', layer = 'cb_2018_us_state_500k')
# world_coastline = st_read('C:/Research_Data/Map_Plot/cb_2018_us_state_500k/cb_2018_us_state_500k.shp', layer = 'cb_2018_us_state_500k')
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
lat_limits = rbind(c(-75, 21), c(-130, 48))
# lat_limits = rbind(c(0, -56), c(0, 80))
# lat_limits_robin = project(xy = as.matrix(lat_limits), proj = coord_info) 
lat_limits_albers = project(xy = as.matrix(lat_limits), proj = coord_info)

## BINN
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
head(current_data_binn_us)

current_data_with_biomes <- current_data_binn_us


#################################################################################
# Plot the traceable parts
#################################################################################
# Plot 1 (a): Scatter plot of carbon input vs total res time, color by biome
plot_1_list = list()
plot_1_a <- ggplot() + 
  geom_point(data = current_data_with_biomes, aes(x = carbon_input, y = total_res_time, color = NA_L1NAME), alpha = 1, size = 1) +
  # geom_point(data = current_data_with_biomes, aes(x = carbon_input, y = total_res_time, color = WWF_MHTNAM), alpha = 1, size = 1) +
  scale_color_manual(values = biome_colors) +
  # theme_minimal() +
  # theme(legend.position = "right") +
  custom_theme +
  labs(x = "Carbon influx (g C m^-2 year^-1)", y = "Ecosystem C residence time (Year)")
  # ggtitle('BINN: Carbon Input vs Total Res Time') +
  # theme(plot.title = element_text(hjust = 0.5))

# plot_1_list[[1]] = plot_1_a


# Plot 1 (b): Adding hyperbolic curves (constant storage capacity) and average values for each biome
# Calculate the average carbon input and total residence time for each biome
biome_averages <- current_data_with_biomes %>%
  group_by(NA_L1NAME) %>%
  # group_by(WWF_MHTNAM) %>%
  summarize(
    avg_carbon_input = mean(carbon_input, na.rm = TRUE),
    avg_total_res_time = mean(total_res_time, na.rm = TRUE)
  ) %>%
  filter(!is.na(avg_carbon_input) & !is.na(avg_total_res_time))

# Create a grid of carbon input and residence time values for the contours
carbon_input_range <- seq(min(biome_averages$avg_carbon_input, na.rm = TRUE) - 200, 
                          max(biome_averages$avg_carbon_input, na.rm = TRUE) + 200, length.out = 100)
total_res_time_range <- seq(min(biome_averages$avg_total_res_time, na.rm = TRUE) - 200, 
                            max(biome_averages$avg_total_res_time, na.rm = TRUE) + 1000, length.out = 100)

# Create a data frame with all combinations of carbon input and residence time
grid_data <- expand.grid(carbon_input = carbon_input_range, total_res_time = total_res_time_range)
grid_data$storage_capacity <- grid_data$carbon_input * grid_data$total_res_time

# Define levels of storage capacity for contour plotting
# contour_levels <- seq(10000, max(grid_data$storage_capacity, na.rm = TRUE), by = 10000)
contour_levels <- seq(10000, 50000, by = 10000)

# Calculate the storage capacity for each grid point
plot_1_b <- ggplot() +
  geom_point(data = biome_averages, aes(x = avg_carbon_input, y = avg_total_res_time, color = NA_L1NAME), size = 20) +
  # geom_point(data = biome_averages, aes(x = avg_carbon_input, y = avg_total_res_time, color = WWF_MHTNAM), size = 4, alpha=1) +
  geom_contour(data = grid_data, aes(x = carbon_input, y = total_res_time, z = storage_capacity), breaks = contour_levels, color = "grey", alpha = 0.8, bins = 10) +
  geom_text_contour(data = grid_data, aes(x = carbon_input, y = total_res_time, z = storage_capacity), 
                    breaks = contour_levels, color = "grey", size = 10, stroke = 0.2, alpha = 0.8, 
                    # label.placer = label_placer_flattest(),  
                    label.placer = label_placer_n(n = 1), 
                    label_format = function(x) paste0(x / 10000, "k")) +
  scale_color_manual(values = biome_colors) + 
  custom_theme +
  scale_x_continuous(limit = c(0, 1100), expand = c(0, 0)) +
  scale_y_continuous(limit = c(1, 350), expand = c(0, 0)) +
  # Overlap theme to remove the legend
  theme(legend.position = "none") +
  labs(x = "Carbon influx (g C m^-2 year^-1)", y = "Ecosystem C residence time (Year)") + 
  theme(plot.margin = unit(c(2, 2, 2, 2), 'inch'))

# plot_1_list[[2]] = plot_1_b
plot_1_list[[1]] = plot_1_b

# Plot 2: 3D plot showing the average values between total_res_time, bulk_xi, and total_res_time_base within each biome
# Average data accross each biome
plot_data_avg <- current_data_with_biomes %>%
  group_by(NA_L1NAME) %>%
  # group_by(WWF_MHTNAM) %>%
  summarize(
    avg_total_res_time = mean(total_res_time, na.rm = TRUE),
    avg_bulk_xi = mean(bulk_xi, na.rm = TRUE),
    avg_res_time_base = mean(total_res_time_base, na.rm = TRUE)
  ) %>%
  filter(!is.na(avg_total_res_time) & !is.na(avg_bulk_xi) & !is.na(avg_res_time_base))

# Create a 2D plot showing the average environmental scaler and the baseline residence time
# Create a grid of environmental scaler and baseline residence time values for the contours
bulk_xi_range <- seq(0.015, # min(plot_data_avg$avg_bulk_xi, na.rm = TRUE)*0.5 
                          max(plot_data_avg$avg_bulk_xi, na.rm = TRUE)*1.5, length.out = 100)
baseline_res_time_range <- seq(min(plot_data_avg$avg_res_time_base, na.rm = TRUE)*0.5, 
                          max(plot_data_avg$avg_res_time_base, na.rm = TRUE)*1.5, length.out = 100)

# Create a data frame with all combinations of carbon input and residence time
grid_data <- expand.grid(bulk_xi = bulk_xi_range, baseline_res_time = baseline_res_time_range)
grid_data$res_time <- grid_data$baseline_res_time / grid_data$bulk_xi

# Define levels of storage capacity for contour plotting
contour_levels <- seq(50, max(grid_data$res_time, na.rm = TRUE), by = 50)
# contour_levels <- seq(10000, 50000, by = 10000)

# Calculate the storage capacity for each grid point
plot_1_c <- ggplot() +
  geom_point(data = plot_data_avg, aes(x = avg_bulk_xi, y = avg_res_time_base, color = NA_L1NAME), size = 20) +
  # geom_point(data = biome_averages, aes(x = avg_carbon_input, y = avg_total_res_time, color = WWF_MHTNAM), size = 4, alpha=1) +
  geom_contour(data = grid_data, aes(x = bulk_xi, y = baseline_res_time, z = res_time), breaks = contour_levels, color = "grey", alpha = 0.8) +
  geom_text_contour(data = grid_data, aes(x = bulk_xi, y = baseline_res_time, z = res_time), 
                    breaks = contour_levels, color = "grey", size = 10, stroke = 0.2, alpha = 0.8, 
                    label.placer = label_placer_flattest(),  
                    label_format = function(x) paste0(x / 10000, "k")) +
  scale_color_manual(values = biome_colors) + 
  custom_theme +
  scale_x_continuous(limit = c(0.02, 0.3), expand = c(0, 0)) +
  scale_y_continuous(limit = c(7, 14), expand = c(0, 0)) +
  labs(x = "Environmental scalar (ξ)", y = "Baseline C residence time (Year)") +
  theme(legend.position = "none") + 
  theme(plot.margin = unit(c(2, 2, 2, 2), 'inch'))

# plot_1_list[[2]] = plot_1_b
plot_1_list[[2]] = plot_1_c

# # Extract the legend from one of the plots
# legend <- get_legend(
#   ggplot() + 
#     geom_point(data = current_data_with_biomes, aes(x = carbon_input, y = total_res_time, color = NA_L1NAME)) +
#     scale_color_manual(values = biome_colors) +
#     custom_theme +
#     theme(legend.position = "right")
# )

top_row <- plot_grid(plot_1_list[[1]], plot_1_list[[2]], ncol = 2, labels = c('(a)', '(b)'), label_size = 70, label_x = 0, label_y = 0.1)

# combined_plot <- plot_grid(
#   plotlist = plot_1_list, ncol = 2, labels = c('a', 'b'), label_size = 30, rel_widths = c(3, 3, 3, 0.10), 
#   # label_x = 0.05, label_y = 1.05,
#   label_x = 0, label_y = 1,
#   label_fontfamily = 'Arial',
#   label_fontface = 'bold'
# )

# Plot biomes on the map with color coding for each biome
biomes_map_plot = ggplot() +
  geom_tile(data = current_data_with_biomes, aes(x = lon, y = lat, fill = NA_L1NAME), height = 60000, width = 60000, na.rm = TRUE) +
  scale_fill_manual(values = biome_colors) +
  geom_sf(data = world_coastline, color = "black", fill = NA, linewidth = 1) +
  coord_sf(xlim = lat_limits_albers[ , 1], ylim = lat_limits_albers[ , 2], datum = NA) +
  # theme(legend.justification = c(0, 0), legend.position = c(-0.01, -0.05), legend.background = element_rect(fill = NA), legend.text.align = 0) +
  # change the size of colorbar
  # guides(fill = guide_colorbar(direction = 'vertical', barwidth = 2, barheight = 10, title.position = 'top', title.hjust = 0, label.hjust = 0, frame.linewidth = 0), reverse = FALSE) +
  theme(legend.text = element_text(size = 40, ), legend.title = element_text(size = 0)) +
  # Add margin between the items in the legend
  theme(legend.key.size = unit(1, 'in'), legend.key.spacing.y = unit(0.5, 'in')) +
  # Add margin between the legend and the plot
  theme(legend.margin = unit(c(0, 0, 0, 2), 'in')) +
  labs(fill = 'Biomes') +
  # add title
  labs(title = paste('BINN Biomes Map'), x = '', y = '') +
  # modify the position of title
  theme(plot.title = element_text(hjust = 0.5, vjust = -1, size = 0)) + 
  # modify the font size
  theme(axis.title = element_text(size = 20)) + 
  theme(panel.background = element_rect(fill = NA, colour = NA)) +
  # modify the margin
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  theme(plot.margin = unit(c(0, 1, 0, 0), 'inch')) +
  theme(axis.text=element_text(size = 35, color = 'black'))

# # Save the plot
# jpeg(paste(cross_validation_dir_output, 'BINN_Biomes_Map.jpg', sep = ''), width = 18, height = 12, units = 'in', res = 300)
# print(biomes_map_plot)
# dev.off()


final_plot <- plot_grid(top_row, biomes_map_plot, ncol = 1, rel_widths = c(2, 0.4), labels = c('','(c)'), label_size = 70, label_x = 0, label_y = 0.1)

# Save the plots
jpeg(paste(cross_validation_dir_output, 'BINN_Traceable_Parts_Carbon_Input_vs_Residence_Time.jpg', sep = ''), width = 50, height = 40, units = 'in', res = 300)
# print(plot_grid(plotlist = plot_1_list, ncol = 2, labels = c('a', 'b'), label_size = 30, rel_widths = c(3, 3, 3, 0.10), 
#           # label_x = 0.05, label_y = 1.05,
#           label_x = 0, label_y = 1,
#           label_fontfamily = 'Arial',
#           label_fontface = 'bold'
# ))
print(final_plot)
dev.off()






# Create a 3D scatter plot using plotly
plot_avg <- plot_ly(
  data = plot_data_avg,
  x = ~avg_bulk_xi,             # Environmental scalar
  y = ~avg_res_time_base,  # Baseline C residence time
  z = ~avg_total_res_time,       # Ecosystem C residence time
  color = ~NA_L1NAME,        # Biome
  # color = ~WWF_MHTNAM,        # Biome
  colors = biome_colors,
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 8, opacity = 1)
) %>%
  layout(
    scene = list(
      xaxis = list(title = "Environmental scalar (ξ)"),
      yaxis = list(title = "Baseline C residence time (Year)"),
      zaxis = list(title = "Ecosystem C residence time (Year)"),
      camera = list(eye = list(x = 1.25, y = 1.25, z = 1.25))
    ),
    legend = list(title = list(text = "Biomes")),
    margin = list(l = 0, r = 0, b = 0, t = 0)
  )

# Display the plot
plot_avg



# Plot 3: 3D plot showing the relationship between total_res_time, bulk_xi, and total_res_time_base within each biome
# Filter data to remove NAs
plot_data_all <- current_data_with_biomes %>%
  filter(!is.na(total_res_time) & !is.na(bulk_xi) & !is.na(total_res_time_base))

# Create a 3D scatter plot using plotly
plot_all <- plot_ly(
  data = plot_data_all,
  x = ~bulk_xi,             # Environmental scalar
  y = ~total_res_time_base,  # Baseline C residence time
  z = ~total_res_time,       # Ecosystem C residence time
  color = ~NA_L1NAME,        # Biome
  # color = ~WWF_MHTNAM,        # Biome
  colors = biome_colors,
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 4, opacity = 0.8)
) %>%
  layout(
    scene = list(
      xaxis = list(title = "Environmental scalar (ξ)"),
      yaxis = list(title = "Baseline C residence time (Year)"),
      zaxis = list(title = "Ecosystem C residence time (Year)"),
      camera = list(eye = list(x = 1.25, y = 1.25, z = 1.25))
    ),
    legend = list(title = list(text = "Biomes")),
    margin = list(l = 0, r = 0, b = 0, t = 0)
  )

# Display the plot
plot_all

