# sensitivity test for CLM5 parameters (21) accross the Continental US
# Import the required libraries
import csv
import functools
import math
import sys
import time
import random
import warnings
import subprocess
import argparse

sys.path.append('/glade/work/haodixu/BINN')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import psutil
import gc

from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

print("Start binns_DDP")

import os
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from multiprocessing import Process


from scipy.io import loadmat
import netCDF4 as ncread 
import mat73

from matplotlib import pyplot as plt
from collections import OrderedDict

#####################################
# Import Different Versions of CLM5 #
#####################################

from fun_matrix_clm5_vectorized import fun_model_simu
import visualization_utils
from fun_matrix_clm5_vectorized_sensitivity import fun_model_sensitivity

################################################
# Command-line arguments
################################################
parser = argparse.ArgumentParser()
parser.add_argument("--par_split", type=int, default=1)
args = parser.parse_args()

# Random seed
random_seed = 12355

# @joshuafan: Set random seeds to try to ensure reproducibility
random.seed(random_seed)
np.random.seed(random_seed) # set the random seed of numpy
torch.manual_seed(random_seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
	dev = 'cuda'
else:
	dev = 'cpu'
# @joshuafan changed
# dev = 'cpu'
device = torch.device(dev) 
print(datetime.now(), '------------device: ', device, '------------')

# print the number of cores
cpu_count = multiprocessing.cpu_count()
thread_count = torch.get_num_threads()
print("Number of Cores: ", cpu_count)
print("Number of threads: ", thread_count)
print(datetime.now(), '------------number of cores: ', cpu_count, '------------')

print(datetime.now(), '------------all packages loaded------------')

# time_stamp = f'{datetime.date(datetime.now())}'
time_stamp = str(datetime.now()).replace(':', '_').replace(' ', '_').replace('.', '_')
job_begin_time = time.time()



################################################
# input data
################################################
cesm2_case_name = 'sasu_f05_g16_checked_step4'
start_year = 661
end_year = 680

time_domain = 'whole_time' # 'whole_time', 'before_1985', 'after_1985', 'random_half_1', 'random_half_2'
model_name = 'cesm2_clm5_cen_vr_v2'

start_id = 1
end_id = 5000
is_resubmit = 0

# pathway
# server path
data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
data_dir_output = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/Sensitivity_Test_1000_norm/'
data_dir_w_scaling = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/Sensitivity_Test_1000_norm/w_scaling/'
os.makedirs(data_dir_output, exist_ok=True)
os.makedirs(data_dir_w_scaling, exist_ok=True)

################################
## Sensitivity Test Constants ##
################################
sensitivity_test_num = 1000


# constants
month_num = 12 
soil_cpool_num = 7
soil_decom_num = 20

#-------------------------------
# wosis data
#-------------------------------
# load wosis data

# The site information for each SOC profile. 
# Names for each column are "profile_id" "country_id" "country_name" "lon" "lat" "layer_num" “date”. 
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc') # wosis profile info
wosis_profile_info = nc_data_middle['soc_profile_info'][:].data.transpose()
nc_data_middle.close()

# The full dataset which contains SOC content information at each layer
# layer_info: "profile_id, date, upper_depth, lower_depth, node_depth, soc_layer_weight, soc_stock, bulk_denstiy, is_pedo"
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_data_integrate_wosis_2019_snapshot_hugelius_mishra.nc') # wosis SOC info
wosis_soc_info = nc_data_middle['data_soc_integrate'][:].data.transpose()
nc_data_middle.close()

#-------------------------------
# PRODA Predicted Parameters
#-------------------------------
# load PRODA predicted parameters

# The site information for each parameter prediction.
# Get the profile id for predicted parameters
# data from nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_1.csv to 9
for i in range(1, 10):
	# contains one column of profile id
	nn_site_loc_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv')
	# contains the predicted parameters (21) for each profile
	nn_site_para_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_para_result_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv')
	# create a dataframe to store the profile id and the parameters
	if i == 1:
		# initialize the dataframe
		PRODA_para = pd.DataFrame(nn_site_loc_temp)
		# rename the column
		PRODA_para.columns = ['profile_id']
		# add the parameters
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
	else:
		# add the parameters
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
# end
# Get the mean value for each parameter for each profile
for i in range(1,22):
	PRODA_para['mean_' + str(i)] = PRODA_para.iloc[:, i:21*10:21].mean(axis = 1)
# end
# Drop the original columns
PRODA_para = PRODA_para.drop(PRODA_para.columns[1:21*9], axis = 1)
# print the head of the dataframe
print(PRODA_para.head())


#-------------------------------
# CLM5 constants
#-------------------------------
# Define parameter names
para_name = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
# Define the prior range for each parameter in the order of para_name
# Original V matrix method
prior_range = [[3*1e-5, 5*1e-4], [3*1e-5, 16*1e-4], [1.2, 3], [0.1, 1], [1, 6], [0.0001, 0.11], [0.1, 0.3], [0.0001, 0.5], [1, 10], [20, 400], [0.1, 0.8], [0.2, 0.8], [0.2, 0.8], [0.0001, 0.4], [0.0001, 0.1], [0.1, 0.74], [0.0001, 0.1], [0.0001, 0.9], [0.5, 1], [0.0001, 5], [0.1, 0.9999]]
# New V matrix method
# prior_range = [[-3, 0], [-10, -6], [1.2, 3], [0.1, 1], [1, 6], [0.0001, 0.11], [0.1, 0.3], [0.0001, 0.5], [1, 10], [20, 400], [0.1, 0.8], [0.2, 0.8], [0.2, 0.8], [0.0001, 0.4], [0.0001, 0.1], [0.1, 0.74], [0.0001, 0.1], [0.0001, 0.9], [0.5, 1], [0.0001, 5], [0.1, 0.9]]
# soil depths info
# width between two interfaces
dz = np.array([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
0.200000000000000, 0.240000000000000, 0.280000000000000, \
0.320000000000000, 0.360000000000000, 0.400000000000000, \
0.440000000000000, 0.540000000000000, 0.640000000000000, \
0.740000000000000, 0.840000000000000, 0.940000000000000, \
1.04000000000000, 1.14000000000000, 2.39000000000000, \
4.67553390593274, 7.63519052838329, 11.1400000000000, \
15.1154248593737])

# depth of the interface
zisoi = np.array([2.000000000000000E-002, 6.000000000000000E-002, \
0.120000000000000, 0.200000000000000, 0.320000000000000, \
0.480000000000000, 0.680000000000000, 0.920000000000000, \
1.20000000000000, 1.52000000000000, 1.88000000000000, \
2.28000000000000, 2.72000000000000, 3.26000000000000, \
3.90000000000000, 4.64000000000000, 5.48000000000000, \
6.42000000000000, 7.46000000000000, 8.60000000000000, \
10.9900000000000, 15.6655339059327, 23.3007244343160, \
34.4407244343160, 49.5561492936897])

zisoi_0 = 0

# depth of the node
zsoi = np.array([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
0.160000000000000, 0.260000000000000, 0.400000000000000, \
0.580000000000000, 0.800000000000000, 1.06000000000000, \
1.36000000000000, 1.70000000000000, 2.08000000000000, \
2.50000000000000, 2.99000000000000, 3.58000000000000, \
4.27000000000000, 5.06000000000000, 5.95000000000000, \
6.94000000000000, 8.03000000000000, 9.79500000000000, \
13.3277669529664, 19.4831291701244, 28.8707244343160, \
41.9984368640029])

# depth between two node
dz_node = zsoi - np.append(np.array([0]), zsoi[:-1], axis = 0)


# cesm2 resolution
cesm2_resolution_lat = 180/384
cesm2_resolution_lon = 360/576
lon_grid = np.arange((-180 + cesm2_resolution_lon/2), 180, cesm2_resolution_lon)
lat_grid = np.arange((90 - cesm2_resolution_lat/2), -90, -cesm2_resolution_lat)

# load cesm2 input
var_name_list = ['nbedrock', 'ALTMAX', 'ALTMAX_LASTYEAR', 'CELLSAND', 'NPP', \
	'SOILPSI', 'TSOI', \
	'W_SCALAR', 'T_SCALAR', 'O_SCALAR', 'FPI_vr', \
	'LITR1_INPUT_ACC_VECTOR', 'LITR2_INPUT_ACC_VECTOR', 'LITR3_INPUT_ACC_VECTOR', 'CWD_INPUT_ACC_VECTOR', \
	'TOTSOMC']

var_name_list_rename =  ['cesm2_simu_nbedrock', 'cesm2_simu_altmax', 'cesm2_simu_altmax_last_year', 'cesm2_simu_cellsand', 'cesm2_simu_npp', \
	'cesm2_simu_soil_water_potnetial', 'cesm2_simu_soil_temperature', \
	'cesm2_simu_w_scalar', 'cesm2_simu_t_scalar', 'cesm2_simu_o_scalar', 'cesm2_simu_n_scalar', \
	'cesm2_simu_input_vector_litter1', 'cesm2_simu_input_vector_litter2', 'cesm2_simu_input_vector_litter3', 'cesm2_simu_input_vector_cwd', \
	'cesm2_simu_soc_stock']

for ivar in np.arange(0, len(var_name_list)):
	# load simulation from CESM2
	var_record_monthly_mean = mat73.loadmat(data_dir_input + 'cesm2_simu/spinup_ss/' + cesm2_case_name + '_cesm2_ss_4da_' + str(start_year) + '_' + str(end_year) + '_' + var_name_list[ivar] + '.mat')
	var_record_monthly_mean = var_record_monthly_mean['var_record_monthly_mean']
	exec(var_name_list_rename[ivar] + ' = var_record_monthly_mean')
# end

for ilayer in np.arange(0, soil_decom_num):
	cesm2_simu_input_vector_litter1[:, :, ilayer, :] = cesm2_simu_input_vector_litter1[:, :, ilayer, :]*dz[ilayer]
	cesm2_simu_input_vector_litter2[:, :, ilayer, :] = cesm2_simu_input_vector_litter2[:, :, ilayer, :]*dz[ilayer]
	cesm2_simu_input_vector_litter3[:, :, ilayer, :] = cesm2_simu_input_vector_litter3[:, :, ilayer, :]*dz[ilayer]
	cesm2_simu_input_vector_cwd[:, :, ilayer, :] = cesm2_simu_input_vector_cwd[:, :, ilayer, :]*dz[ilayer]
#end

cesm2_simu_input_sum_litter1 = np.sum(cesm2_simu_input_vector_litter1, axis = 2)
cesm2_simu_input_sum_litter2 = np.sum(cesm2_simu_input_vector_litter2, axis = 2)
cesm2_simu_input_sum_litter3 = np.sum(cesm2_simu_input_vector_litter3, axis = 2)
cesm2_simu_input_sum_cwd = np.sum(cesm2_simu_input_vector_cwd, axis = 2)

del cesm2_simu_input_vector_litter1, cesm2_simu_input_vector_litter2, cesm2_simu_input_vector_litter3, cesm2_simu_input_vector_cwd

# representative points 
sample_profile_id = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_representative_profiles.mat')
sample_profile_id = sample_profile_id['sample_profile_id']
# convert the number to be starting from 0 in python world
sample_profile_id = sample_profile_id - 1

# choose the profile id with lat and lon within the range of the United States
profile_collection = np.where(
    (wosis_profile_info[:, 2] == 156) & 
	(wosis_profile_info[:, 3] >= -124.763068) & 
    (wosis_profile_info[:, 3] <= -66.949895) & 
    (wosis_profile_info[:, 4] >= 24.5) & 
    (wosis_profile_info[:, 4] <= 49.384358)
)[0]

################################
# If use same dataset as PRODA #
################################
# load mat file
para_gr = loadmat(data_dir_input + 'wosis_2019_snap_shot/cesm2_clm5_cen_vr_v2_para_gr.mat')
stat_r2 = loadmat(data_dir_input + 'wosis_2019_snap_shot/cesm2_clm5_cen_vr_v2_stat_r2.mat')
eligible_profile = loadmat(data_dir_input + 'wosis_2019_snap_shot/eligible_profile_loc_0_cesm2_clm5_cen_vr_v2_whole_time.mat')
para_gr = para_gr['para_gr']
stat_r2 = stat_r2['stat_r2']
eligible_profile = eligible_profile['eligible_loc_0']
# convert the number to be starting from 0 in python world
eligible_profile = eligible_profile - 1
# calculate average value per row in para_gr, and choose those profiles with average value less than 1.05
# calculate average value per row in stat_r2, and choose those profiles with average value larger than 0
# choose profile that listed in eligible_profile
PRODA_collection = np.where((np.mean(para_gr, axis = 1) < 1.05) & 
							(np.mean(stat_r2, axis = 1) > 0) & 
							(np.isin(np.arange(0, wosis_profile_info.shape[0]), eligible_profile) == True) & 
							# Also in the column profile_id of the dataframe PRODA_para
							(np.isin(np.arange(0, wosis_profile_info.shape[0]), PRODA_para['profile_id']) == True)
							)[0]
# Choose overlap between profile_collection and PRODA_collection
profile_collection = np.intersect1d(profile_collection, PRODA_collection)

# Choose random 2000 profiles for testing
profile_collection = np.random.choice(profile_collection, 512, replace=False)

###############################################################################################################

profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])

profile_range = np.arange(0, len(profile_collection))

print('number of profiles: ', len(profile_collection))

print(datetime.now(), '------------all input data loaded------------')


#---------------------------------------------------
# wrap up soc data for NN
#---------------------------------------------------
obs_soc_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each non-nan column is an SOC observation
obs_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the depth of the corresponding SOC observation in "obs_soc_matrix"
obs_upper_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the upper depth of the corresponding SOC observation in "obs_soc_matrix"
obs_lower_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the lower depth of the corresponding SOC observation in "obs_soc_matrix"
obs_lon_lat_loc = np.ones([len(profile_collection), 2])*np.nan

model_force_input_vector_cwd = np.ones([len(profile_collection), month_num])*np.nan
model_force_input_vector_litter1 = np.ones([len(profile_collection), month_num])*np.nan
model_force_input_vector_litter2 = np.ones([len(profile_collection), month_num])*np.nan
model_force_input_vector_litter3 = np.ones([len(profile_collection), month_num])*np.nan

model_force_altmax_lastyear_profile = np.ones([len(profile_collection), month_num])*np.nan
model_force_altmax_current_profile = np.ones([len(profile_collection), month_num])*np.nan
model_force_nbedrock = np.ones([len(profile_collection), month_num])*np.nan

model_force_xio = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan
model_force_xin = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan

model_force_sand_vector = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan

model_force_soil_temp_profile = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan
model_force_soil_water_profile = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan

# record the sum of recorded layers for all profiles
layer_num_record = 0

for iprofile_hat in profile_range:
	# profile num
	iprofile = profile_collection[iprofile_hat]
	# profile id
	profile_id = wosis_profile_info[iprofile, 0]
	# find currently using profile
	loc_profile = np.where(wosis_soc_info[:, 0] == profile_id)[0]
	# find the lon and lat info of soil profile
	lon_profile = wosis_profile_info[iprofile, 3]
	lat_profile = wosis_profile_info[iprofile, 4]
	
	lat_loc = np.where(abs(lat_profile - lat_grid) == min(abs(lat_profile - lat_grid)))[0][0]
	lon_loc = np.where(abs(lon_profile - lon_grid) == min(abs(lon_profile - lon_grid)))[0][0]
	
	# info of the node depth of profile  
	wosis_layer_depth = wosis_soc_info[loc_profile, 4]
	# observed C info (gC/m3)
	wosis_layer_obs = wosis_soc_info[loc_profile, 6]
	# check how many layers are recorded
	layer_num_record = layer_num_record + len(wosis_layer_obs)
	# observced upper depth of each layer
	wosis_layer_upper_depth = wosis_soc_info[loc_profile, 2]
	# observced lower depth of each layer
	wosis_layer_lower_depth = wosis_soc_info[loc_profile, 3]
	# exclude nan values
	valid_soc_loc = np.where((np.isnan(wosis_layer_obs) == False) & (np.isnan(wosis_layer_depth) == False) & (np.isnan(wosis_layer_upper_depth) == False) & (np.isnan(wosis_layer_lower_depth) == False))
	# valid layer number
	num_layers = len(valid_soc_loc[0])
	
	if num_layers > 0:
		wosis_layer_depth = wosis_layer_depth[valid_soc_loc]/100 # convert unit from cm to m
		wosis_layer_obs = wosis_layer_obs[valid_soc_loc]
		wosis_layer_upper_depth = wosis_layer_upper_depth[valid_soc_loc]/100
		wosis_layer_lower_depth = wosis_layer_lower_depth[valid_soc_loc]/100
		
		obs_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_depth
		obs_soc_matrix[iprofile_hat, 0:num_layers] = wosis_layer_obs
		obs_upper_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_upper_depth
		obs_lower_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_lower_depth

	# end if num_layers > 0:

	obs_lon_lat_loc[iprofile_hat, :] = [lon_loc, lat_loc]
	
	# input vector
	model_force_input_vector_cwd[iprofile_hat, :] = cesm2_simu_input_sum_cwd[lat_loc, lon_loc, :]
	model_force_input_vector_litter1[iprofile_hat, :] = cesm2_simu_input_sum_litter1[lat_loc, lon_loc, :]
	model_force_input_vector_litter2[iprofile_hat, :] = cesm2_simu_input_sum_litter2[lat_loc, lon_loc, :]
	model_force_input_vector_litter3[iprofile_hat, :] = cesm2_simu_input_sum_litter3[lat_loc, lon_loc, :]
	# altmax current and last year
	model_force_altmax_lastyear_profile[iprofile_hat, :] = cesm2_simu_altmax_last_year[lat_loc, lon_loc, :]
	model_force_altmax_current_profile[iprofile_hat, :] = cesm2_simu_altmax[lat_loc, lon_loc, :]
	# nbedrock
	model_force_nbedrock[iprofile_hat, :] = cesm2_simu_nbedrock[lat_loc, lon_loc, :]
	# oxygen scalar
	model_force_xio[iprofile_hat, :, :] = cesm2_simu_o_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	# nitrogen scalar
	model_force_xin[iprofile_hat, :, :] = cesm2_simu_n_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	# sand content
	model_force_sand_vector[iprofile_hat, :, :] = cesm2_simu_cellsand[lat_loc, lon_loc, 0:soil_decom_num, :]
	# soil temperature and water potential
	model_force_soil_temp_profile[iprofile_hat, :, :] = cesm2_simu_soil_temperature[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_soil_water_profile[iprofile_hat, :, :] = cesm2_simu_w_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	
# end
# check the overall number of layers in the profile
print("Number of layers in profile: " + str(layer_num_record))
print(datetime.now(), '------------soc data prepared------------')
########################################################
# neural network (BINNS)
########################################################
nn_split_ratio = 0.1
test_split_ratio = 0.1
#---------------------------------------------------
# env info
#---------------------------------------------------
# environmental info of soil profiles

env_info_names = ['ProfileNum', 'ProfileID', 'LayerNum', 'Lon', 'Lat', 'Date', \
'Rmean', 'Rmax', 'Rmin', \
'ESA_Land_Cover', \
'ET', \
'IGBP', 'Climate', 'Soil_Type', 'NPPmean', 'NPPmax', 'NPPmin', \
'Veg_Cover', \
'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
'Abs_Depth_to_Bedrock', \
'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm',\
'CEC_0cm', 'CEC_30cm', 'CEC_100cm', \
'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', \
'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', \
'Depth_Bedrock_R', \
'Garde_Acid', \
'Occurrence_R_Horizon', \
'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', \
'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', \
'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', \
'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', \
'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', \
'USDA_Suborder', \
'WRB_Subgroup', \
'Drought', \
'Elevation', \
'Max_Depth', \
'Koppen_Climate_2018', \
'cesm2_npp', 'cesm2_npp_std', \
'cesm2_gpp', 'cesm2_gpp_std', \
'cesm2_vegc', \
'nbedrock', \
'R_Squared']

categorical_vars = [['ESA_Land_Cover'], ['Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm'], 
					['USDA_Suborder'], ['WRB_Subgroup'], ['Koppen_Climate_2018']]  # Variables inside a sub-list share the same categories
categorical_vars_flattened = [item for sublist in categorical_vars for item in sublist]

env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info.mat')
env_info = env_info['EnvInfo']
original_lons = env_info[:, 3].copy()
original_lats = env_info[:, 4].copy()

col_max_min = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present_cesm2_clm5_cen_vr_v2_whole_time_col_max_min.mat')
col_max_min = col_max_min['col_max_min']

# Don't want to transform categorical variables, so set max/min to nan
for group in categorical_vars:
	for var in group:
		idx = env_info_names.index(var)
		# print("Var {} Nans {}".format(var, np.count_nonzero(np.isnan(env_info[:, idx]))))
		col_max_min[idx, :] = np.nan

# warnings.filterwarnings("error")
for ivar in np.arange(3, len(col_max_min[:, 0])):
	if np.isnan(col_max_min[ivar, :]).any():
		pass
	else:
		env_info[:, ivar] = (env_info[:, ivar] - col_max_min[ivar, 0])/(col_max_min[ivar, 1] - col_max_min[ivar, 0])
		env_info[(env_info[:, ivar] > 1), ivar] = 1
		env_info[(env_info[:, ivar] < 0), ivar] = 0
	# except:
	# 	print('error in variable: ', ivar)
# warnings.resetwarnings()

env_info = df(env_info)

# env_info_scaled = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_' + model_name  + '_' + time_domain + '_maxmin_scaled.mat')
# env_info_scaled = df(env_info_scaled['profile_env_info'])
# env_info = env_info_scaled



env_info.columns = env_info_names
env_info["original_lon"] = original_lons
env_info["original_lat"] = original_lats

# # @joshuafan added temporarily
# env_info.index = env_info.ProfileNum
# print("Env info old shape", env_info.shape)
# print("Env info", env_info.head())
# print(profile_collection[0:5, 0])

# variables used in training the NN
var4nn = ['Lon', 'Lat', \
'ESA_Land_Cover', \
# 'IGBP', \
# 'Climate', \
# 'Soil_Type', \
# 'NPPmean', 'NPPmax', 'NPPmin', \
# 'Veg_Cover', \
'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
'Abs_Depth_to_Bedrock', \
'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm',\
'CEC_0cm', 'CEC_30cm', 'CEC_100cm', \
'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', \
'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', \
# 'Depth_Bedrock_R', \
'Garde_Acid', \
'Occurrence_R_Horizon', \
'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', \
'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', \
'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', \
'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', \
'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', \
'USDA_Suborder', \
'WRB_Subgroup', \
# 'Drought', \
'Elevation', \
# 'Max_Depth', \
'Koppen_Climate_2018', \
'cesm2_npp', 'cesm2_npp_std', \
# 'cesm2_gpp', 'cesm2_gpp_std', \
'cesm2_vegc', \
'nbedrock']


#---------------------------------------------------
# training data
#---------------------------------------------------
current_data_x = np.ones((len(profile_collection), 60, 12, 13))*np.nan
current_data_x[:, 0:60, 0, 0] = np.array(env_info.loc[profile_collection[:, 0], var4nn])
current_data_x[:, 0:12, 0, 1] = model_force_input_vector_cwd
current_data_x[:, 0:12, 0, 2] = model_force_input_vector_litter1
current_data_x[:, 0:12, 0, 3] = model_force_input_vector_litter2
current_data_x[:, 0:12, 0, 4] = model_force_input_vector_litter3
current_data_x[:, 0:12, 0, 5] = model_force_altmax_lastyear_profile
current_data_x[:, 0:12, 0, 6] = model_force_altmax_current_profile
current_data_x[:, 0:12, 0, 7] = model_force_nbedrock

current_data_x[:, 0:20, 0:12, 8] = model_force_xio
current_data_x[:, 0:20, 0:12, 9] = model_force_xin
current_data_x[:, 0:20, 0:12, 10] = model_force_sand_vector
current_data_x[:, 0:20, 0:12, 11] = model_force_soil_temp_profile
current_data_x[:, 0:20, 0:12, 12] = model_force_soil_water_profile


current_data_y = obs_soc_matrix
current_data_z = obs_depth_matrix


lons = np.array(env_info.loc[profile_collection[:, 0], "original_lon"])
lats = np.array(env_info.loc[profile_collection[:, 0], "original_lat"])


nan_loc = np.nanmean(current_data_y, axis = 1) + \
			np.sum(current_data_x[:, 0:60, 0, 0], axis = 1) + \
			np.sum(model_force_input_vector_cwd, axis = 1) + \
			np.sum(model_force_input_vector_litter1, axis = 1) + \
			np.sum(model_force_input_vector_litter2, axis = 1) + \
			np.sum(model_force_input_vector_litter3, axis = 1) + \
			np.sum(model_force_altmax_lastyear_profile, axis = 1) + \
			np.sum(model_force_altmax_current_profile, axis = 1) + \
			np.sum(model_force_nbedrock, axis = 1) + \
			np.sum(model_force_xio, axis = (1, 2)) + \
			np.sum(model_force_xin, axis = (1, 2)) + \
			np.sum(model_force_sand_vector, axis = (1, 2)) + \
			np.sum(model_force_soil_temp_profile, axis = (1, 2)) + \
			np.sum(model_force_soil_water_profile, axis = (1, 2))

valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0] ### Why change the shape from 26915 to 26934??? ###

current_data_y = current_data_y[valid_profile_loc, :]
current_data_z = current_data_z[valid_profile_loc, :]
current_data_x = current_data_x[valid_profile_loc, :, :, :]
current_data_profile_id = profile_collection[valid_profile_loc, 0]
obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]
print("Shape of current data x", current_data_x.shape)
print("Shape of current data y", current_data_y.shape)
print("Shape of current data z", current_data_z.shape)
print("Shape of obs upper depth matrix", obs_upper_depth_matrix.shape)
print("Shape of obs lower depth matrix", obs_lower_depth_matrix.shape)
# env_info = env_info.loc[valid_profile_loc, :]

# Select PRODA parameters so that the Profile_IDs match the current data
PRODA_para = PRODA_para.loc[PRODA_para['profile_id'].isin(current_data_profile_id)]
PRODA_para = PRODA_para.sort_values(by='profile_id')                 
print("Shape of PRODA para", PRODA_para.shape)

print(datetime.now(), '------------parameters loaded------------')

# Helper function to combine the training data into a single tensor
class MergeDataset(Dataset):
    def __init__(self, data_x, data_y, data_z, profile_id):
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.profile_id = profile_id

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx], self.data_z[idx], self.profile_id[idx]
	
# Start training
def worker(rank, world_size):

	start_time = time.time()

	# Initialize the process group
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(world_size)
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

	# Initialize distributed environment
	dist.init_process_group('gloo', rank=rank, world_size=world_size, timeout=timedelta(hours=1))

	# Initialize datasets
	dataset = MergeDataset(current_data_x, current_data_y, current_data_z, current_data_profile_id)

	# Use DistributedSampler for distributed training
	dist_sampler = DistributedSampler(dataset, shuffle=True)

	# Data loaders with DistributedSampler
	dist_loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=dist_sampler)

	# Idx number
	ibatch = 0
	num_singular = 0

	# Initialze the tensor to store the sensitivity
	sensitivity_all_rank = torch.zeros([len(dist_loader), 3], requires_grad=False, device=device)
	sensitivity_0_30_rank = torch.zeros([len(dist_loader), 3], requires_grad=False, device=device)
	sensitivity_30_100_rank = torch.zeros([len(dist_loader), 3], requires_grad=False, device=device)
	sensitivity_100__rank = torch.zeros([len(dist_loader), 3], requires_grad=False, device=device)
	sensitivity_profile_id = torch.zeros([len(dist_loader)], requires_grad=False, device=device)

	#####################
	# Observed Variable #
	#####################

	# For each profile, calculate the observed SOC
	for batch_info in dist_loader:
		batch_x, batch_y, batch_z, batch_profile_id = batch_info
		sensitivity_profile_id[ibatch] = batch_profile_id
		# Initialize the parameter list
		obs_para = torch.zeros([len(para_name)], requires_grad=False, device=device)
		# Initialize the tensor to store the observed SOC
		obs_soc_all = torch.zeros([1000], requires_grad=False, device=device)
		obs_soc_0_30 = torch.zeros([1000], requires_grad=False, device=device)
		obs_soc_30_100 = torch.zeros([1000], requires_grad=False, device=device)
		obs_soc_100_ = torch.zeros([1000], requires_grad=False, device=device)	
		for idx_obs in range(1000):
			# Freely select parameters from prior range
			for ipara in range(len(para_name)):
				# obs_para[ipara] = torch.tensor(np.random.uniform(prior_range[ipara][0], prior_range[ipara][1]), requires_grad=False, device=device)
				obs_para[ipara] = torch.tensor(np.random.normal(((prior_range[ipara][0] + prior_range[ipara][1])/2), \
					(prior_range[ipara][1] - prior_range[ipara][0])/6), requires_grad=False, device=device)
			# end for parameter selection
			
			# Initialize the model
			temp_obs_soc, whether_singular_obs = fun_model_sensitivity(obs_para, batch_x)
			if whether_singular_obs:
				# print('Singular matrix at profile ', batch_profile_id.item())
				num_singular += 1
				obs_soc_all[idx_obs] = torch.nan
				obs_soc_0_30[idx_obs] = torch.nan
				obs_soc_30_100[idx_obs] = torch.nan
				obs_soc_100_[idx_obs] = torch.nan
				# skip current loop
				continue
			obs_soc_all[idx_obs] = torch.nansum(temp_obs_soc[0, :])
			obs_soc_0_30[idx_obs] = torch.nansum(temp_obs_soc[0, 0:6])
			obs_soc_30_100[idx_obs] = torch.nansum(temp_obs_soc[0, 6:9])
			obs_soc_100_[idx_obs] = torch.nansum(temp_obs_soc[0, 9:])

		# Remove nan and inf
		obs_soc_all = obs_soc_all[~torch.isnan(obs_soc_all) & ~torch.isinf(obs_soc_all)]
		obs_soc_0_30 = obs_soc_0_30[~torch.isnan(obs_soc_0_30) & ~torch.isinf(obs_soc_0_30)]
		obs_soc_30_100 = obs_soc_30_100[~torch.isnan(obs_soc_30_100) & ~torch.isinf(obs_soc_30_100)]
		obs_soc_100_ = obs_soc_100_[~torch.isnan(obs_soc_100_) & ~torch.isinf(obs_soc_100_)]

		# Calculate the variance of the observed SOC
		obs_soc_all_var = torch.var(obs_soc_all)
		obs_soc_0_30_var = torch.var(obs_soc_0_30)
		obs_soc_30_100_var = torch.var(obs_soc_30_100)
		obs_soc_100_var = torch.var(obs_soc_100_)

		# Check if the observed SOC is nan or inf
		if torch.isnan(obs_soc_all_var) or torch.isnan(obs_soc_0_30_var) \
			or torch.isnan(obs_soc_30_100_var) or torch.isnan(obs_soc_100_var):
			
			print('Observed SOC variance is nan at profile ', batch_profile_id.item())
			print("Parameters: ", obs_para)

		if torch.isinf(obs_soc_all_var) or torch.isinf(obs_soc_0_30_var) \
			or torch.isinf(obs_soc_30_100_var) or torch.isinf(obs_soc_100_var):
			print('Observed SOC variance is inf at profile ', batch_profile_id.item())
			print("Parameters: ", obs_para)
		
		if rank == 0:
			# Save the observed SOC to a file
			pd.DataFrame(obs_soc_all.cpu().numpy()).to_csv(data_dir_output + 'obs_soc_all.csv', index=False)
			print("Batch number: ", ibatch)
			# Print the variance of the observed SOC
			print('Variance of the observed SOC at all layers: ', obs_soc_all_var)
			print('Variance of the observed SOC at 0-30cm: ', obs_soc_0_30_var)
			print('Variance of the observed SOC at 30-100cm: ', obs_soc_30_100_var)
			print('Variance of the observed SOC at 100-: ', obs_soc_100_var)
			print('Observed SOC variance calculated in ', time.time() - start_time, ' seconds')

		print("Number of singular matrix and SOC larger than 5000 kgC/m2 encounter in profile ", batch_profile_id.item(), " is ", num_singular)

	
		####################
		# Sensitivity Test #
		####################
		# initialize the parameter list
		test_para = obs_para.clone().detach()
		num_singular_test = 0
		# Initialize the tensor to store the sensitivity
		sensitivity_soc_all = torch.zeros([3, 100], requires_grad=False, device=device)
		sensitivity_soc_0_30 = torch.zeros([3, 100], requires_grad=False, device=device)
		sensitivity_soc_30_100 = torch.zeros([3, 100], requires_grad=False, device=device)
		sensitivity_soc_100_ = torch.zeros([3, 100], requires_grad=False, device=device)
		# Define parameter range for sensitivity test based on parsor input
		if args.par_split == 1:
			start_para_idx = 0
			end_para_idx = 3
		elif args.par_split == 2:
			start_para_idx = 3
			end_para_idx = 6
		elif args.par_split == 3:
			start_para_idx = 6
			end_para_idx = 9
		elif args.par_split == 4:
			start_para_idx = 9
			end_para_idx = 12
		elif args.par_split == 5:
			start_para_idx = 12
			end_para_idx = 15
		elif args.par_split == 6:
			start_para_idx = 15
			end_para_idx = 18
		elif args.par_split == 7:
			start_para_idx = 18
			end_para_idx = 21
		else:
			raise ValueError('Invalid par_split input')
		# Loop through all parameters
		for ipara in range(start_para_idx, end_para_idx):

			para_cal_start_time = time.time()

			for idx_test in range(100):
				# Initialize the parameter for sensitivity test
				# test_para[ipara] = torch.tensor(np.random.uniform(prior_range[ipara][0], prior_range[ipara][1]), requires_grad=False, device=device)
				test_para[ipara] = torch.tensor(np.random.normal(((prior_range[ipara][0] + prior_range[ipara][1])/2), \
					(prior_range[ipara][1] - prior_range[ipara][0])/6), requires_grad=False, device=device)
				
				# Initialize the tensor to store the simulated SOC
				temp_soc_all = torch.zeros([sensitivity_test_num], requires_grad=False, device=device)
				temp_soc_0_30 = torch.zeros([sensitivity_test_num], requires_grad=False, device=device)
				temp_soc_30_100 = torch.zeros([sensitivity_test_num], requires_grad=False, device=device)
				temp_soc_100_ = torch.zeros([sensitivity_test_num], requires_grad=False, device=device)

				# loop through other parameters to calculate the sensitivity
				for idx_temp in range(sensitivity_test_num):

					# Freely select parameters from prior range
					for ipara_temp in range(len(para_name)):
						if ipara_temp != ipara:
							# test_para[ipara_temp] = torch.tensor(np.random.uniform(prior_range[ipara_temp][0], prior_range[ipara_temp][1]), requires_grad=False, device=device)
							test_para[ipara_temp] = torch.tensor(np.random.normal(((prior_range[ipara_temp][0] + prior_range[ipara_temp][1])/2), \
								(prior_range[ipara_temp][1] - prior_range[ipara_temp][0])/6), requires_grad=False, device=device)
						# end if ipara_temp != ipara
					# end for ipara_temp in range(len(para_name))
					
					# Initialize the model
					temp_soc, whether_singular_test = fun_model_sensitivity(test_para, batch_x)
					if whether_singular_test:
						num_singular_test += 1
						temp_soc_all[idx_temp] = torch.nan
						temp_soc_0_30[idx_temp] = torch.nan
						temp_soc_30_100[idx_temp] = torch.nan
						temp_soc_100_[idx_temp] = torch.nan
						# skip current loop
						continue
					temp_soc_all[idx_temp] = torch.nansum(temp_soc[0, :])
					temp_soc_0_30[idx_temp] = torch.nansum(temp_soc[0, 0:6])
					temp_soc_30_100[idx_temp] = torch.nansum(temp_soc[0, 6:9])
					temp_soc_100_[idx_temp] = torch.nansum(temp_soc[0, 9:])
				# end for idx_temp in range(100)
					
				# Save the results to files for parameter 20 w-scaling
				if ipara == 20:
					pd.DataFrame(temp_soc_all.cpu().numpy()).to_csv(data_dir_w_scaling + 'temp_soc_all' + '_' + str(rank) + '_' + str(batch_profile_id.item()) + '_' + str(idx_test) + '.csv', index=False)
					pd.DataFrame(temp_soc_0_30.cpu().numpy()).to_csv(data_dir_w_scaling + 'temp_soc_0_30' + '_' + str(rank) + '_' + str(batch_profile_id.item()) + '_' + str(idx_test) + '.csv', index=False)
					pd.DataFrame(temp_soc_30_100.cpu().numpy()).to_csv(data_dir_w_scaling + 'temp_soc_30_100' + '_' + str(rank) + '_' + str(batch_profile_id.item()) + '_' + str(idx_test) + '.csv', index=False)
					pd.DataFrame(temp_soc_100_.cpu().numpy()).to_csv(data_dir_w_scaling + 'temp_soc_100_' + '_' + str(rank) + '_' + str(batch_profile_id.item()) + '_' + str(idx_test) + '.csv', index=False)
				# end if ipara == 20
									
				# Calculate the mean of the simulated SOC and store it
				sensitivity_soc_all[ipara-(args.par_split-1)*3, idx_test] = torch.nanmean(temp_soc_all)
				sensitivity_soc_0_30[ipara-(args.par_split-1)*3, idx_test] = torch.nanmean(temp_soc_0_30)
				sensitivity_soc_30_100[ipara-(args.par_split-1)*3, idx_test] = torch.nanmean(temp_soc_30_100)
				sensitivity_soc_100_[ipara-(args.par_split-1)*3, idx_test] = torch.nanmean(temp_soc_100_)

				# check if the simulated SOC is nan or inf
				if torch.isnan(sensitivity_soc_all[ipara-(args.par_split-1)*3, idx_test]) or torch.isnan(sensitivity_soc_0_30[ipara-(args.par_split-1)*3, idx_test]) \
						or torch.isnan(sensitivity_soc_30_100[ipara-(args.par_split-1)*3, idx_test]) or torch.isnan(sensitivity_soc_100_[ipara-(args.par_split-1)*3, idx_test]):
					print('Simulated SOC is nan at profile ', batch_profile_id.item())
					print("Parameters: ", test_para)

				if torch.isinf(sensitivity_soc_all[ipara-(args.par_split-1)*3, idx_test]) or torch.isinf(sensitivity_soc_0_30[ipara-(args.par_split-1)*3, idx_test]) \
						or torch.isinf(sensitivity_soc_30_100[ipara-(args.par_split-1)*3, idx_test]) or torch.isinf(sensitivity_soc_100_[ipara-(args.par_split-1)*3, idx_test]):
					print('Simulated SOC is inf at profile ', batch_profile_id.item())
					print("Parameters: ", test_para)
			
			if rank == 0:
				print('Parameter ', para_name[ipara], ' soc calculated in ', time.time() - para_cal_start_time, ' seconds')
			# end for idx_test in range(100)

		# end for ipara in range(len(para_name))
			
		# Gather the soc from all processes
		# Reshape the sensitivity from [21, 100] to [100, 21] first
		sensitivity_soc_all = sensitivity_soc_all.T
		sensitivity_soc_0_30 = sensitivity_soc_0_30.T
		sensitivity_soc_30_100 = sensitivity_soc_30_100.T
		sensitivity_soc_100_ = sensitivity_soc_100_.T

		# Calculate the variance of the sensitivity
		# Shape after var: [3]
		sensitivity_soc_all_var = torch.var(sensitivity_soc_all, dim=0) / obs_soc_all_var
		sensitivity_soc_0_30_var = torch.var(sensitivity_soc_0_30, dim=0) / obs_soc_0_30_var
		sensitivity_soc_30_100_var = torch.var(sensitivity_soc_30_100, dim=0) / obs_soc_30_100_var
		sensitivity_soc_100_var = torch.var(sensitivity_soc_100_, dim=0) / obs_soc_100_var

		# Save the sensitivity to the tensor
		# Shape after save: [len(dist_loader), 3]
		sensitivity_all_rank[ibatch, :] = sensitivity_soc_all_var
		sensitivity_0_30_rank[ibatch, :] = sensitivity_soc_0_30_var
		sensitivity_30_100_rank[ibatch, :] = sensitivity_soc_30_100_var
		sensitivity_100__rank[ibatch, :] = sensitivity_soc_100_var

		if rank == 0:
			print('Sensitivity for batch ', ibatch, ' calculated in ', time.time() - start_time, ' seconds')
		
		ibatch += 1

	# Reshape the sensitivity to a 1D tensor
	# Shape after reshape: [1, len(dist_loader), 3]
	sensitivity_all_rank = sensitivity_all_rank.unsqueeze(0)
	sensitivity_0_30_rank = sensitivity_0_30_rank.unsqueeze(0)
	sensitivity_30_100_rank = sensitivity_30_100_rank.unsqueeze(0)
	sensitivity_100__rank = sensitivity_100__rank.unsqueeze(0)
	# Reshape the profile id to a 1D tensor
	# Shape after reshape: [1, len(dist_loader)]
	sensitivity_profile_id = sensitivity_profile_id.unsqueeze(0)

	# Gather the sensitivity from all processes
	# Shape after gather: [world_size, len(dist_loader), 3]
	sensitivity_all_gather = [torch.zeros_like(sensitivity_all_rank) for _ in range(world_size)]
	sensitivity_0_30_gather = [torch.zeros_like(sensitivity_0_30_rank) for _ in range(world_size)]
	sensitivity_30_100_gather = [torch.zeros_like(sensitivity_30_100_rank) for _ in range(world_size)]
	sensitivity_100__gather = [torch.zeros_like(sensitivity_100__rank) for _ in range(world_size)]
	dist.all_gather(sensitivity_all_gather, sensitivity_all_rank)
	dist.all_gather(sensitivity_0_30_gather, sensitivity_0_30_rank)
	dist.all_gather(sensitivity_30_100_gather, sensitivity_30_100_rank)
	dist.all_gather(sensitivity_100__gather, sensitivity_100__rank)
	# Gather the profile id from all processes
	# Shape after gather: [world_size, len(dist_loader)]
	sensitivity_profile_id_gather = [torch.zeros_like(sensitivity_profile_id) for _ in range(world_size)]
	dist.all_gather(sensitivity_profile_id_gather, sensitivity_profile_id)

	# Cat and reshape the sensitivity to the shape [len(dist_loader)*world_size, 3]
	sensitivity_all_var = torch.cat(sensitivity_all_gather, dim=0).reshape(-1, 3)
	sensitivity_0_30_var = torch.cat(sensitivity_0_30_gather, dim=0).reshape(-1, 3)
	sensitivity_30_100_var = torch.cat(sensitivity_30_100_gather, dim=0).reshape(-1, 3)
	sensitivity_100_var = torch.cat(sensitivity_100__gather, dim=0).reshape(-1, 3)
	# Cat and reshape the profile id to the shape [len(dist_loader)*world_size]
	sensitivity_profile_id = torch.cat(sensitivity_profile_id_gather, dim=0).reshape(-1)

	# Calculate the mean of the sensitivity
	# Shape after mean: [3]
	sensitivity_soc_all_var_mean = torch.nanmean(sensitivity_all_var, dim=0)
	sensitivity_soc_0_30_var_mean = torch.nanmean(sensitivity_0_30_var, dim=0)
	sensitivity_soc_30_100_var_mean = torch.nanmean(sensitivity_30_100_var, dim=0)
	sensitivity_soc_100_var_mean = torch.nanmean(sensitivity_100_var, dim=0)

	
	# Save the sensitivity to excel file
	if rank == 0:
		pd.DataFrame(sensitivity_all_var.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_all_sensitivity_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_0_30_var.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_0_30_sensitivity_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_30_100_var.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_30_100_sensitivity_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_100_var.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_100__sensitivity_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_soc_all_var_mean.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_all_sensitivity_mean_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_soc_0_30_var_mean.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_0_30_sensitivity_mean_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_soc_30_100_var_mean.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_30_100_sensitivity_mean_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_soc_100_var_mean.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_soc_100__sensitivity_mean_' + str(args.par_split) + '.csv')
		pd.DataFrame(sensitivity_profile_id.cpu().numpy()).to_csv(data_dir_output + '/sensitivity_profile_id_' + str(args.par_split) + '.csv')
		print('Sensitivity calculated in ', time.time() - start_time, ' seconds')
	
	print("Number of singular matrix and SOC larger than 5000 kgC/m2 encounter in profile ", sensitivity_profile_id.item(), " is ", num_singular_test)

# Initialize the process
if __name__ == '__main__':
	# Number of CPUs requester
	world_size = 128
	processes = []
	for rank in range(world_size):
		p = Process(target=worker, args=(rank, world_size))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()