"""
Main training script for BINN, using DistributedDataParallel.

See run_binn.sh for example usage with reasonable hyperparameters.
"""
import csv
import functools
import math
import sys
import time
import random
import warnings
import subprocess
import argparse
import misc_utils
from mlp import mlp_wrapper
from sklearn.model_selection import KFold

from mlp import mlp_wrapper, nn_only
from pe_gcn_model import GridCellSpatialRelationEncoder
from spatial_utils import *
from losses import binns_loss
import visualization_utils

sys.path.append('/glade/work/haodixu/BINN')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

print("Start binns_DDP")

# Set default dtype to float64 to avoid underflow in process-based model.
torch.set_default_dtype(torch.float64)

# Temporary hack to avoid printing np.float64(...) when printing out numpy scalars.
# TODO fix this
np.set_printoptions(legacy="1.25")

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from multiprocessing import Process
from scipy.io import loadmat
import netCDF4 as ncread 
import mat73
from matplotlib import pyplot as plt

###################################
# Import CLM5 process-based model #
###################################
# fun_model_simu predicts at user-specified depths. fun_model_prediction predicts at 20 default layers.
from fun_matrix_clm5_vectorized import fun_model_simu, fun_model_prediction

# fun_bulk_simu returns additional components (quantities describing physical processes)
from fun_matrix_clm5_vectorized_bulk_converge import fun_bulk_simu

################################################
# Command-line arguments
################################################
parser = argparse.ArgumentParser()

# Model architecture
parser.add_argument("--note", type=str, default="", help="Optional name to give to the model")
parser.add_argument("--model", type=str, default="old_mlp", choices=['old_mlp', 'new_mlp', 'lipmlp', 'nn_only'], help="Model type")
parser.add_argument("--width", type=int, default=128, help="Size of hidden layers (new_mlp or nn_only)")
parser.add_argument("--categorical", type=str, default="embedding", choices=["embedding", "one_hot"], help="How to embed categorical variables")
parser.add_argument("--embed_dim", type=int, default=5, help="Embedding dim for each categorical variable (if using embeddings)")
parser.add_argument("--use_bn", action='store_true', help="Whether to use batchnorm")
parser.add_argument("--dropout_prob", default=0., type=float, help="Dropout prob")
parser.add_argument("--activation", type=str, choices=['relu', 'leaky_relu', 'tanh'], default='relu', help="Activation function inside neural network")

# Process-based model settings
parser.add_argument("--vertical_mixing", type=str, default='original', choices=['original', 'simple_one_intercept', 'simple_two_intercepts'], help="""Vertical mixing matrix parameterization. Original explicitly models diffusion.
						 simple_one_intercept approximates with a log-log relationship with depth (upwards/downwards
						 having the same intercept). simple_two_intercepts allows upwards/downwards transfers to
						 have different intercepts.""")
parser.add_argument("--vectorized", type=str, default='yes', choices=['yes', 'no', 'compare'], help="""yes to use vectorized version of process-based model,
						 no to use old for-loop version, compare to run both and assert they produce the same result""")
parser.add_argument("--para_to_predict", type=str, default="all", choices=["all", "four", "fifteen"], help="Which parameters to predict using NN. If 'four', the NN only predicts the four most sensitive parameters, and other parameters are prescribed to PRODA-predicted values.")

# Sigmoid temp and initialization
parser.add_argument("--min_temp", type=float, default=10., help="Min temp for sigmoid")
parser.add_argument("--max_temp", type=float, default=109., help="Max temp for sigmoid")
parser.add_argument("--init", type=str, default="xavier_uniform", choices=["default", "xavier_uniform", "kaiming_uniform"], help="Initialization for weights. For xavier_uniform/kaiming_uniform, biases are initialized to zero.")

# Data split
parser.add_argument("--data_seed", type=int, default=-1, help="Random seed for splitting data. -1 means use same as args.seed")
parser.add_argument("--n_datapoints", type=int, default=-1, help="Set this to train on a random subset of this many datapoints (for train+val+test). -1 to use the whole dataset")
parser.add_argument("--representative_sample", action='store_true', help="Whether to restrict to ~1000 'representative profiles'")
parser.add_argument("--cross_val_idx", type=int, default=0, help="""Cross-validation index. 0 means no cross-validation (fixed train/val/test split), 
                    		while a number between 1 and k means to use that index's fold (1-based). Note that this script only runs one fold;
							you need to manually combine results from multiple folds.""")
parser.add_argument("--n_folds", type=int, default=10, help="Number of folds if using cross-validation")
parser.add_argument("--split", type=str, default='random', choices=['random', 'horizontal', 'vertical', 'us_vs_world', 'grid2'],
					help="""How to split val/test sets. If `random`, just hold out random examples (cross-validation or fixed split).
						 If horizontal or vertical, split into 10 horizontal or vertical folds; only cross-validation is supported.
						 If us_vs_world, use US as train/val sets and rest-of-world as test set; cross-validation is not supported (only fixed split).""")
parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of datapoints in validation set. Only used if not doing cross-validation.")
parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of datapoints in test set. Only used if not doing cross-validation.")
parser.add_argument("--synthetic_labels", action='store_true', help="Whether to use synthetic SOC labels (generated from running CLM5 on PRODA parameters)")

# Transformations
parser.add_argument("--standardize_input", action='store_true', help="If set, standardize numeric features to mean 0, std 1. Otherwise, features vary between 0 and 1.")
parser.add_argument("--standardize_output", action='store_true', help="ONLY APPLIES IF MODEL IS NN_ONLY. If set, standardize outputs (labels) to mean 0, std 1.")

# Training
parser.add_argument("--seed", type=int, default=0, help="Random seed for model initialization")
parser.add_argument("--optimizer", type=str, choices=["SGD", "AdamW"], default="AdamW")
parser.add_argument("--scheduler", type=str, choices=["none", "reduce_on_plateau", "step", "cosine"], default="none")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (SGD ONLY)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--save_freq", type=int, default=5, help="How often (epochs) to save the latest checkpoint, in case the job crashes")

# Regularization
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--clip_value", type=float, default=-1, help="Clip value for gradient clipping. -1 for no clipping.")

# Losses and weights
parser.add_argument("--losses", nargs="+", choices=["l1", "smooth_l1", "l2", "param_reg", "spectral", "lipmlp"], default=["smooth_l1", "param_reg"],
					help="Losses to use (can list any number).")
parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 10.0], help="Weights for each loss, in the same order as `args.losses`")

# Positional encoding
parser.add_argument("--lonlat_features", action='store_true', help="Whether longitude and latitude should be passed as features")
parser.add_argument("--pos_enc", type=str, default='none', choices=['none', 'early', 'late'],
					help="How lon/lat features are encoded. 'none' means not used. 'early' means that positional encoding is concatenated with other features. 'late' means that it is only used as an error term for the latent parameters.")

# Computational environment
parser.add_argument("--use_ddp", type=int, default=1, help="Whether to use DDP")
parser.add_argument("--num_CPU", type=int, default=32, help="Number of processes for Torch DDP")
parser.add_argument("--job_scheduler", type=str, default="pbs", choices=["pbs", "slurm"], help="Job scheduler system. Job scheduler. PBS for NCAR computers, slurm for AIDA server.")
parser.add_argument("--whether_resume", type=int, default=0, help="Whether to resume training from a previous model")
parser.add_argument("--previous_job_id", type=str, default="", help="Previous job id to resume from (otherwise, resumes using envir variable PREVIOUS_JOB_ID)")
parser.add_argument("--time_limit", type=float, default=11.5, help="Time limit for this job in HOURS. If trainng is not finished yet, start another job to continue.")

args = parser.parse_args()

# Make sure the correct number of loss weights were provided
assert(len(args.losses) == len(args.lambdas))
args.lambdas = torch.tensor(args.lambdas)


def set_seeds(seed):
	"""
	Attempts to set all random seeds to improve reproducibility.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

# Set seeds for reproducibility
set_seeds(args.seed)
if args.data_seed == -1:
	args.data_seed = args.seed

# Keep track of when the job started
job_begin_time = time.time()


#################################################################################
# ATTENTION: There is a lot of code that is not inside any function,
# which sets up datasets. On CPU (start_method=fork), this gets run
# ONCE, and the forked processes have access to all variables created.
# On GPU (start_method=spawn), this gets run once initially, and AGAIN
# for each process. It would be nice to factor this out eventually.
################################################################################# 

################################################
# Data Directories (CHANGE THIS!!!)
################################################
# data_dir_input = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/'
# data_dir_output = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/BINNS/OUTPUT_DATA/'
# data_dir_input = 'C:/Users/hx293/Research_Data/BINN/ENSEMBLE/INPUT_DATA/'
# data_dir_output = 'C:/Users/hx293/Unsync_Data/BINN_output/'
# server path
# job_submit_path = '/glade/u/home/haodixu/BINN/PBS_Submit/Bulk_Converge/'
# data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
# data_dir_output = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/'
data_dir_input = '../../ENSEMBLE/INPUT_DATA/'
data_dir_output = '../../OUTPUT_DATA/'
job_submit_path = './resume_jobs/'
os.makedirs(job_submit_path, exist_ok=True)

################################################
# Setup datasets
################################################
cesm2_case_name = 'sasu_f05_g16_checked_step4'
start_year = 661
end_year = 680

time_domain = 'whole_time' # 'whole_time', 'before_1985', 'after_1985', 'random_half_1', 'random_half_2'
model_name = 'cesm2_clm5_cen_vr_v2'

start_id = 1
end_id = 5000
is_resubmit = 0

# constants
month_num = 12 
soil_cpool_num = 7
soil_decom_num = 20

#-------------------------------
# Load wosis data
#-------------------------------
# The site information for each SOC profile. 
# Names for each column are "profile_id" "country_id" "country_name" "lon" "lat" "layer_num" "date".
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc')  # wosis profile info
wosis_profile_info = nc_data_middle['soc_profile_info'][:].data.transpose()
nc_data_middle.close()

# The full dataset which contains SOC content information at each layer
# layer_info: "profile_id, date, upper_depth, lower_depth, node_depth, soc_layer_weight, soc_stock, bulk_denstiy, is_pedo"
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_data_integrate_wosis_2019_snapshot_hugelius_mishra.nc')  # wosis SOC info
wosis_soc_info = nc_data_middle['data_soc_integrate'][:].data.transpose()
nc_data_middle.close()

#--------------------------------
# Load PRODA Predicted Parameters
#--------------------------------
for i in range(1, 10):
	# Loop over all runs (folds) of PRODA. Construct a dataframe where each row
	# is a site. The first column is the profile ID. The next 21 columns are
	# parameters from the first run, next 21 are parameters from the second one, etc.

	# contains one column of profile id
	nn_site_loc_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)
	# contains the predicted parameters (21) for each profile
	nn_site_para_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_para_result_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)

	if i == 1:
		# Initialize the dataframe with just profile_id
		PRODA_para = pd.DataFrame(nn_site_loc_temp)
		PRODA_para.columns = ['profile_id']

		# Concatenate parameters on the right
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
	else:
		# Concatenate this run's parameters on the right
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
# end
# For each parameter at each site, take the average across all runs
for i in range(1,22):
	PRODA_para['mean_' + str(i)] = PRODA_para.iloc[:, i:21*10:21].mean(axis = 1)
# end
# Drop the original columns
PRODA_para = PRODA_para.drop(PRODA_para.columns[1:21*9], axis = 1)

# Convert profile ID to zero-based, to match how WOSIS data is processed below
PRODA_para['profile_id'] = PRODA_para['profile_id'] - 1
print("PRODA parameters:")
print(PRODA_para.head())


#-------------------------------
# CLM5 constants
#-------------------------------
# Parameter names
if args.vertical_mixing == 'original':
	para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
else:
	# If using the simpler vertical mixing parameterization, replace diffus/cryo with slope/intercept.
	para_names = ['slope', 'intercept', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
	if args.vertical_mixing == 'simple_two_intercepts':
		para_names.append('intercept_leach')

# Parameter indices that the neural network predicts. Usually we predict all the parameters, but for
# the retrieval test we may prescribe some and only predict 4 or 15 most sensitive parameters.
if args.para_to_predict == "all":  # All parameters
	para_index = np.arange(0, len(para_names))
elif args.para_to_predict == "four":
	assert len(para_names) == 21
	para_index = np.array([3, 9, 14, 19])
elif args.para_to_predict == "fifteen":
	assert len(para_names) == 21
	para_index = np.array([0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])
else:
	raise ValueError("Invalid value of --para_to_predict")

# Soil depths info
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


############################################
# Select subset of observations (profiles) #
############################################
# Representative points
sample_profile_id = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_representative_profiles.mat')

# NOTE: Not sure why "sample_profile_id" shape is [100, 50] before flattening?
sample_profile_id = sample_profile_id['sample_profile_id'].flatten()

# convert the number to be starting from 0 in python world
sample_profile_id = sample_profile_id - 1

# Choose the profile id with lat and lon within the range of the United States
profile_collection = np.where(
	(wosis_profile_info[:, 2] == 156) & 
	(wosis_profile_info[:, 3] >= -124.763068) & 
	(wosis_profile_info[:, 3] <= -66.949895) & 
	(wosis_profile_info[:, 4] >= 24.5) & 
	(wosis_profile_info[:, 4] <= 49.384358)
)[0]

##################################
# If using same dataset as PRODA #
##################################
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

if args.representative_sample:
	# Restrict to only "representative profiles" (1018)
	profile_collection = np.intersect1d(profile_collection, sample_profile_id)

if args.n_datapoints != -1:
	# Choose random subset of profiles for testing.
	rng = np.random.default_rng(seed=args.data_seed)
	profile_collection = rng.choice(profile_collection, args.n_datapoints, replace=False)

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
obs_lon_lat_loc = np.ones([len(profile_collection), 2])*np.nan  # Each row is a profile. First column is longitude, second column is latitude

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

# Variables used in training the NN.
GEOGRAPHY_VARS = ['Lon', 'Lat', 'Elevation', 'Abs_Depth_to_Bedrock', 'Occurrence_R_Horizon', 'nbedrock']
if not args.lonlat_features:
	GEOGRAPHY_VARS.remove('Lon')
	GEOGRAPHY_VARS.remove('Lat')
CLIMATE_VARS = ['Koppen_Climate_2018', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19']
SOIL_TEXTURE_VARS = ['USDA_Suborder', 'WRB_Subgroup', 'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm',
					 'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', 'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm',
					 'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', 'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm',
					 'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm']
SOIL_CHEMICAL_VARS = ['SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', 'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm',
					  'CEC_0cm', 'CEC_30cm', 'CEC_100cm', 'Garde_Acid']
VEGETATION_VARS = ['ESA_Land_Cover', 'cesm2_npp', 'cesm2_npp_std', 'cesm2_vegc']
var4nn = GEOGRAPHY_VARS + CLIMATE_VARS + SOIL_TEXTURE_VARS + SOIL_CHEMICAL_VARS + VEGETATION_VARS

# Load environmental covariates
env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info.mat')
env_info = env_info['EnvInfo']
original_lons = env_info[:, 3].copy()
original_lats = env_info[:, 4].copy()

# Min/max for each feature
col_max_min = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present_cesm2_clm5_cen_vr_v2_whole_time_col_max_min.mat')
col_max_min = col_max_min['col_max_min']


################################################
# Categorical variables                        #
################################################
# List of categorical variables. Inner lists group categorical
# variables that share the same categories. For example, the category IDs
# in Texture_USDA_0cm, Texture_USDA_30cm have the same semantic meaning,
# so they share an embedding space.
categorical_vars = [['ESA_Land_Cover'], ['Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm'], 
					['USDA_Suborder'], ['WRB_Subgroup'], ['Koppen_Climate_2018']]  # Variables inside a sub-list share the same categories
categorical_vars_flattened = [item for sublist in categorical_vars for item in sublist]


#####################################################################
# Transform covariates to [0, 1] range based on precomputed min/max #
#####################################################################
# Don't want to transform categorical variables, so set max/min to nan
for group in categorical_vars:
	for var in group:
		idx = env_info_names.index(var)
		col_max_min[idx, :] = np.nan

# warnings.filterwarnings("error")
for ivar in np.arange(3, len(col_max_min[:, 0])):
	if np.isnan(col_max_min[ivar, :]).any():
		pass
	else:
		env_info[:, ivar] = (env_info[:, ivar] - col_max_min[ivar, 0])/(col_max_min[ivar, 1] - col_max_min[ivar, 0])
		env_info[(env_info[:, ivar] > 1), ivar] = 1
		env_info[(env_info[:, ivar] < 0), ivar] = 0

env_info = df(env_info)
env_info.columns = env_info_names
env_info["original_lon"] = original_lons
env_info["original_lat"] = original_lats


########################################################################
# Preprocessing of categorical variables. Perhaps this should be moved
# inside the neural network itself to make usage easier.
#########################################################################
# Determine how many indices are in each categorical group
var_to_categories = dict()  # varname to number of categories
for group in categorical_vars:
	n_categories = int(np.nanmax(env_info[group]) + 1)
	for var in group:
		var_to_categories[var] = n_categories
print("Var to categories", var_to_categories)


#---------------------------------------------------
# training data
#---------------------------------------------------
# Input features (environmental covariates)
current_data_x = np.ones((len(profile_collection), len(var4nn), 12, 13))*np.nan

# Fill in input features
# NOTE: env_info is indexed starting from 0, and profile_collection
# is also using zero-based indices
current_data_x[:, 0:len(var4nn), 0, 0] = np.array(env_info.loc[profile_collection[:, 0], var4nn])

# Monthly forcing variables
current_data_x[:, 0:12, 0, 1] = model_force_input_vector_cwd
current_data_x[:, 0:12, 0, 2] = model_force_input_vector_litter1
current_data_x[:, 0:12, 0, 3] = model_force_input_vector_litter2
current_data_x[:, 0:12, 0, 4] = model_force_input_vector_litter3
current_data_x[:, 0:12, 0, 5] = model_force_altmax_lastyear_profile
current_data_x[:, 0:12, 0, 6] = model_force_altmax_current_profile
current_data_x[:, 0:12, 0, 7] = model_force_nbedrock

# Forcing variables that apply for each depth layer
current_data_x[:, 0:20, 0:12, 8] = model_force_xio
current_data_x[:, 0:20, 0:12, 9] = model_force_xin
current_data_x[:, 0:20, 0:12, 10] = model_force_sand_vector
current_data_x[:, 0:20, 0:12, 11] = model_force_soil_temp_profile
current_data_x[:, 0:20, 0:12, 12] = model_force_soil_water_profile

# SOC labels: each row represents a location, each column represents an observation
# at a certain depth. Since locations have different numbers of observations, many entries are NaN.
current_data_y = obs_soc_matrix

# Depth of each observation in current_data_y (same shape)
current_data_z = obs_depth_matrix

# Geographic coordinates
lons = np.array(env_info.loc[profile_collection[:, 0], "original_lon"])
lats = np.array(env_info.loc[profile_collection[:, 0], "original_lat"])
current_data_c = np.stack([lons, lats], axis=1)  # [profile, 2]: lon/lat of each site

# Remove sites with missing features or forcing variables
nan_loc = np.nanmean(current_data_y, axis = 1) + \
			np.sum(current_data_x[:, 0:len(var4nn), 0, 0], axis = 1) + \
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
valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0] ### Why change the shape from 26915 to 26934??? ###  joshuafan: some sites may have missing forcing data or covariates

current_data_y = current_data_y[valid_profile_loc, :]
current_data_z = current_data_z[valid_profile_loc, :]
current_data_x = current_data_x[valid_profile_loc, :, :, :]
current_data_c = current_data_c[valid_profile_loc, :]
current_data_profile_id = profile_collection[valid_profile_loc, 0]
obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]
print("Shape of current data x", current_data_x.shape)
print("Shape of current data y", current_data_y.shape)
print("Shape of current data z", current_data_z.shape)
print("Shape of current data c", current_data_c.shape)
print("Shape of current_data_profile_id", current_data_profile_id.shape)
print("Shape of obs upper depth matrix", obs_upper_depth_matrix.shape)
print("Shape of obs lower depth matrix", obs_lower_depth_matrix.shape)
print("Shape of env info", env_info.shape)


# Select PRODA parameters so that the Profile_IDs match the current data
PRODA_para = PRODA_para.loc[PRODA_para['profile_id'].isin(current_data_profile_id)]
PRODA_para = PRODA_para.sort_values(by='profile_id')

# Store the PRODA_para into numpy array (mean_1 to mean_21)
current_PRODA_para = PRODA_para[['mean_1', 'mean_2', 'mean_3', 'mean_4', 'mean_5', 'mean_6', 'mean_7', 'mean_8', 'mean_9', 'mean_10', 'mean_11', \
								 'mean_12', 'mean_13', 'mean_14', 'mean_15', 'mean_16', 'mean_17', 'mean_18', 'mean_19', 'mean_20', 'mean_21']].to_numpy()
print("Shape of PRODA para", current_PRODA_para.shape)

# Clamp to [0, 1]
current_PRODA_para = np.clip(current_PRODA_para, a_min=0, a_max=1)

#############################
# PRODA soc simulation data #
#############################
# Initialize numpy array for PRODA soc simulation data
PRODA_soc_simu = np.ones((len(current_data_profile_id), 200))*np.nan

if args.synthetic_labels:
	start_time = time.time()
	for i in range(len(current_data_profile_id)):
		# Get the current profile's data
		current_data_x_simu = current_data_x[i, :, :, :]
		current_data_z_simu = current_data_z[i, :]
		current_PRODA_para_simu = current_PRODA_para[i, :]

		# Convert the data to tensor, reshape to shape [1, 60, 12, 13] and [1, 21]
		current_data_x_simu = torch.tensor(current_data_x_simu, dtype=torch.float32).unsqueeze(0)
		current_data_z_simu = torch.tensor(current_data_z_simu, dtype=torch.float32).unsqueeze(0)
		current_PRODA_para_simu = torch.tensor(current_PRODA_para_simu, dtype=torch.float32).unsqueeze(0)

		# Run the simulation
		PRODA_soc_simu[i, :] = fun_model_simu(current_PRODA_para_simu, current_data_x_simu, current_data_z_simu, args.vertical_mixing, args.vectorized)

		# If any simulation is over 1,000,000 gC/m2, set it to nan
		if np.any(PRODA_soc_simu[i, :] > 1000000):
			print(">>>>>>>>>>>>>>>>>>>>>>>> Extreme simulated SOC. Coordinates", current_data_c[i, :])
			print("PRODA params", current_PRODA_para[i, :])
			valid_loc = ~np.isnan(current_data_z[i, :])
			print("Depths", current_data_z[i, valid_loc])
			print("SOC simu", PRODA_soc_simu[i, valid_loc])
			print("SOC obs", current_data_y[i, valid_loc])

	# Drop the profiles with all nan values
	valid_profile_loc = np.where(np.all(np.isnan(PRODA_soc_simu), axis=1) == False)[0]
	current_data_y = current_data_y[valid_profile_loc, :]
	current_data_z = current_data_z[valid_profile_loc, :]
	current_data_c = current_data_c[valid_profile_loc, :]
	current_data_x = current_data_x[valid_profile_loc, :, :, :]
	current_data_profile_id = current_data_profile_id[valid_profile_loc]
	current_PRODA_para = current_PRODA_para[valid_profile_loc, :]
	PRODA_soc_simu = PRODA_soc_simu[valid_profile_loc, :]
	obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
	obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]

	print("Time taken to run PRODA soc simu", time.time() - start_time)
	print("Shape of PRODA soc simu", PRODA_soc_simu.shape)
	print("Shape of current data x", current_data_x.shape)

	# If using synthetic labels, treat the simulated SOC as the true labels
	current_data_y = PRODA_soc_simu


###############################################################
# Load checkpoint if resuming a previous run.
# We do this outside the main function, since the checkpoint
# stores the train/val/test split for setting up the datasets.
##################################################################
# If PREVIOUS_JOB_ID environment variable set, overwrite the
# commandline arg.
if 'PREVIOUS_JOB_ID' in os.environ:
	args.previous_job_id = os.environ.get('PREVIOUS_JOB_ID')
	print("Overrode previous_job_id. Now", args.previous_job_id)

# Load checkpoint if resuming. This checkpoint is only used to read the
# data splits, the actual weights will be loaded later.
if args.whether_resume == 1:
	checkpoint_path = data_dir_output + 'neural_network/' + args.previous_job_id + '/checkpoint_' + args.previous_job_id + '.pt'
	checkpoint_main = torch.load(checkpoint_path, weights_only=False)

	# Delete the job submit file if it exists
	try:
		os.remove(job_submit_path + 'Resume' + args.previous_job_id + '.submit')
	except OSError:
		pass


################################################################
# Data splitting
################################################################
n_datapoints = current_data_x.shape[0]

# Train, validation, test split
if args.whether_resume == 0:
	if args.cross_val_idx == 0:
		# Single train/val/test split, no cross-validation.
		# Compute train/val/test indices
		if args.split == 'random':
			# Randomly split US datapoints into train/val/test (ignoring geography) 
			rng = np.random.default_rng(seed=args.data_seed)
			# Determine the number of training samples based on the ratios
			train_loc = rng.choice(np.arange(0, n_datapoints), size=round((1 - args.val_ratio - args.test_ratio) * n_datapoints), replace=False)
			# The remaining data after removing the training samples
			remaining_loc = np.setdiff1d(np.arange(0, n_datapoints), train_loc)
			# Split the remaining data into validation and test sets
			num_val_samples = round(args.val_ratio / (args.val_ratio + args.test_ratio) * len(remaining_loc))
			val_loc = rng.choice(remaining_loc, size=num_val_samples, replace=False)
			test_loc = np.setdiff1d(remaining_loc, val_loc)
		elif args.split == 'us_vs_world':
			# Train set is southern US, validation set is northern US, test set is rest of world
			train_loc = np.flatnonzero(
				(wosis_profile_info[current_data_profile_id, 2] == 156) &
				(wosis_profile_info[current_data_profile_id, 3] >= -124.763068) &
				(wosis_profile_info[current_data_profile_id, 3] <= -66.949895) &
				(wosis_profile_info[current_data_profile_id, 4] >= 24.5) &
				(wosis_profile_info[current_data_profile_id, 4] <= 40.)
			)
			val_loc = np.flatnonzero(
				(wosis_profile_info[current_data_profile_id, 2] == 156) &
				(wosis_profile_info[current_data_profile_id, 3] >= -124.763068) &
				(wosis_profile_info[current_data_profile_id, 3] <= -66.949895) &
				(wosis_profile_info[current_data_profile_id, 4] > 40.) &
				(wosis_profile_info[current_data_profile_id, 4] <= 49.384358)
			)
			test_loc = np.setdiff1d(np.arange(0, n_datapoints), train_loc)
			test_loc = np.setdiff1d(test_loc, val_loc)
		else:
			raise ValueError("If cross_val_idx is 0, only random or us_vs_world split is supported. To use north/south or east/west splits, set cross_val_idx to a number between 1 and 10.")

	else:
		# Split the data into k-folds, either randomly or by spatial block
		# Recall cross_val_idx is one-based. Subtract one to make it zero-based. The test fold
		# is given by cross_val_idx-1, and validation fold is one larger (cross_val_idx % n_folds)
		test_fold = args.cross_val_idx - 1
		val_fold = args.cross_val_idx % args.n_folds
		if args.split == 'random':
			# Random split
			# Assign the test dataset based on the cross-validation index, and val dataset
			# Randomly split the remaining data into training and validation sets
			kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.data_seed)
			fold_indices = list(kf.split(np.arange(len(current_data_x[:, 0]))))
			test_loc = fold_indices[args.cross_val_idx - 1][1]
			train_val_idx = fold_indices[args.cross_val_idx - 1][0]
			train_loc = np.random.choice(train_val_idx, size=round((1 - nn_split_ratio - test_split_ratio)/(1 - test_split_ratio) * len(train_val_idx)), replace=False)
			val_loc = np.setdiff1d(train_val_idx, train_loc)
		elif args.split == "horizontal":
			# Compute latitude thresholds separating folds. If we have 10 folds, we have 11 boundary thresholds.
			sorted_lats = np.sort(current_data_c[:, 1])   # current_data_c[:, 1] contains latitudess
			indices = np.linspace(0, n_datapoints, num=args.n_folds, endpoint=False)  # Fold start/stop indices: [0, n_datapoints*0.1, n_datapoints*0.2, ... n_datapoints*0.9]
			lat_thresholds = [sorted_lats[int(i)] for i in indices]  # Lat boundaries between folds
			lat_thresholds.append(sorted_lats[-1] + 1)  # Add final threshold above all datapoints

			# Recall cross_val_idx is one-based. Subtract one to make it zero-based. The test fold
			# is given by cross_val_idx-1, and validation fold is one larger (cross_val_idx % n_folds)
			test_loc = np.flatnonzero((current_data_c[:, 1] >= lat_thresholds[test_fold]) & (current_data_c[:, 1] < lat_thresholds[test_fold+1]))
			val_loc = np.flatnonzero((current_data_c[:, 1] >= lat_thresholds[val_fold]) & (current_data_c[:, 1] < lat_thresholds[val_fold+1]))

			# Train loc is all indices except val/test
			train_loc = np.setdiff1d(np.arange(0, n_datapoints), test_loc)
			train_loc = np.setdiff1d(train_loc, val_loc)
		elif args.split == "vertical":
			# Compute longitude thresholds separating folds. If we have 10 folds, we have 11 boundary thresholds.
			sorted_lons = np.sort(current_data_c[:, 0])   # current_data_c[:, 1] contains longitudes
			indices = np.linspace(0, n_datapoints, num=args.n_folds, endpoint=False)  # Fold start/stop indices: [0, n_datapoints*0.1, n_datapoints*0.2, ... n_datapoints*0.9]
			lon_thresholds = [sorted_lons[int(i)] for i in indices]  # Lon boundaries between folds
			lon_thresholds.append(sorted_lons[-1] + 1)  # Add final threshold above all datapoints

			# Recall cross_val_idx is one-based. Subtract one to make it zero-based. The test fold
			# is given by cross_val_idx-1, and validation fold is one larger (cross_val_idx % n_folds)
			test_loc = np.flatnonzero((current_data_c[:, 0] >= lon_thresholds[test_fold]) & (current_data_c[:, 0] < lon_thresholds[test_fold+1]))
			val_loc = np.flatnonzero((current_data_c[:, 0] >= lon_thresholds[val_fold]) & (current_data_c[:, 0] < lon_thresholds[val_fold+1]))

			# Train loc is all indices except val/test
			train_loc = np.setdiff1d(np.arange(0, n_datapoints), test_loc)
			train_loc = np.setdiff1d(train_loc, val_loc)
		elif args.split == "grid2":
			GRID_SIZE = 2  # in degrees longitude/latitude

			# For each site, compute its coordinates in a GRID_SIZE*GRID_SIZE grid
			min_lon, max_lon = current_data_c[:, 0].min(), current_data_c[:, 0].max()

			# Nearest multiple of GRID_SIZE below min_lon (https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python)
			min_lon_rounded = GRID_SIZE * np.floor(min_lon / GRID_SIZE)
			max_lon_rounded = GRID_SIZE * np.ceil(max_lon / GRID_SIZE)
			col_idx = np.floor((current_data_c[:, 0] - min_lon_rounded) / GRID_SIZE)
			n_cols = round((max_lon_rounded - min_lon_rounded) / GRID_SIZE)

			# Repeat for lat
			min_lat, max_lat = current_data_c[:, 1].min(), current_data_c[:, 1].max()
			min_lat_rounded = GRID_SIZE * np.floor(min_lat / GRID_SIZE)
			max_lat_rounded = GRID_SIZE * np.ceil(max_lat / GRID_SIZE)
			row_idx = np.floor((current_data_c[:, 1] - min_lat_rounded) / GRID_SIZE)
			n_rows = round((max_lat_rounded - min_lat_rounded) / GRID_SIZE)

			# Compute a "grid cell ID"
			cell_id = (row_idx * n_cols + col_idx).astype(int)

			# Split cells into folds. See https://stackoverflow.com/questions/33398017/to-generate-a-split-indices-for-n-fold
			s = np.arange(n_rows * n_cols)
			random.Random(args.data_seed).shuffle(s)
			val_cells = s[val_fold::args.n_folds]
			test_cells = s[test_fold::args.n_folds]

			# Split sites
			test_loc = np.flatnonzero(np.isin(cell_id, test_cells))
			val_loc = np.flatnonzero(np.isin(cell_id, val_cells))
			train_loc = np.setdiff1d(np.setdiff1d(np.arange(0, n_datapoints), test_loc), val_loc)
			for i in range(args.n_folds):
				fold_cells = s[i::args.n_folds]
				print("Fold", i, "Examples", len(np.flatnonzero(np.isin(cell_id, fold_cells))))
		else:
			raise ValueError("Invalid value of --split")

else:
	# If we are resuming, load the same train/val/test indices.
	train_loc = checkpoint_main['train_indices']
	val_loc = checkpoint_main['val_indices']
	test_loc = checkpoint_main['test_indices']

# Construct the train/val/test splits
train_y = torch.tensor(current_data_y[train_loc, :],)
val_y = torch.tensor(current_data_y[val_loc, :])
test_y = torch.tensor(current_data_y[test_loc, :])

train_z = torch.tensor(current_data_z[train_loc, :])
val_z = torch.tensor(current_data_z[val_loc, :])
test_z = torch.tensor(current_data_z[test_loc, :])

train_c = torch.tensor(current_data_c[train_loc, :])
val_c = torch.tensor(current_data_c[val_loc, :])
test_c = torch.tensor(current_data_c[test_loc, :])

train_x = torch.tensor(current_data_x[train_loc, :, :, :])
val_x = torch.tensor(current_data_x[val_loc, :, :, :])
test_x = torch.tensor(current_data_x[test_loc, :, :, :])

train_profile_id = torch.tensor(current_data_profile_id[train_loc])
val_profile_id = torch.tensor(current_data_profile_id[val_loc])
test_profile_id = torch.tensor(current_data_profile_id[test_loc])

train_proda_para = torch.tensor(current_PRODA_para[train_loc, :])
val_proda_para = torch.tensor(current_PRODA_para[val_loc, :])
test_proda_para = torch.tensor(current_PRODA_para[test_loc, :])

print("Shape of train data", train_x.shape)
print("Shape of val data", val_x.shape)
print("Shape of test data", test_x.shape)
print(datetime.now(), '------------nn data prepared------------')

#---------------------------------------------------
# Grid env info for prediction
#---------------------------------------------------
# load grid env info
grid_env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present.mat')
grid_env_info = grid_env_info['EnvInfo']
original_lons_grid = grid_env_info[:, 0].copy()
original_lats_grid = grid_env_info[:, 1].copy()

# column names
# environmental info of global grids 
# Difference: does not include first 3 columns 'ProfileNum', 'ProfileID', 'LayerNum' and the last column 'R_Squared'
# Therefore, we choose to use the original categorical column names 
grid_env_info_names = [\
	'Lon', 'Lat', 'Date', \
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
	'nbedrock']

# Remove the first 3 columns and the last column from the col_max_min matrix
col_max_min_grid = col_max_min[3:-1, :]

# Normalize grid env info
for ivar in np.arange(0, len(col_max_min_grid[:, 0])):
	if np.isnan(col_max_min_grid[ivar, :]).any():
		pass
	else:
		grid_env_info[:, ivar] = (grid_env_info[:, ivar] - col_max_min_grid[ivar, 0])/(col_max_min_grid[ivar, 1] - col_max_min_grid[ivar, 0])
		grid_env_info[(grid_env_info[:, ivar] > 1), ivar] = 1
		grid_env_info[(grid_env_info[:, ivar] < 0), ivar] = 0

grid_env_info = df(grid_env_info)
grid_env_info.columns = grid_env_info_names

# Only keep the variables used in training the NN
grid_env_info = grid_env_info[var4nn]
grid_env_info["original_lon"] = original_lons_grid
grid_env_info["original_lat"] = original_lats_grid

# Exclude all rows with nan values
grid_env_info = grid_env_info.dropna(axis=0, how='any')

# Select the rows with lon and lat values within continental US
grid_US_mask = (grid_env_info["original_lon"] >= -124.763068) \
			& (grid_env_info["original_lon"] <= -66.949895) \
			& (grid_env_info["original_lat"] >= 24.521694) \
			& (grid_env_info["original_lat"] <= 49.384358)  # True if grid cell is within US bounding box
grid_US_profiles = np.where(grid_US_mask)[0]  # Indices (zero-based 'grid profile IDs') of grid cells in US, used later
grid_env_info_US = grid_env_info[grid_US_mask]
grid_env_info_num = grid_env_info_US.shape[0]
print("Shape of grid env info (after dropping nans, selecting US):", grid_env_info_US.shape)


# Check the max value of categorical variables, if it is larger than the number of categories, then remove the row
for group in categorical_vars:
	mask = grid_env_info_US[group].apply(lambda x: (x > np.max(env_info[group])).any(), axis=1)
	indices_to_remove = grid_env_info_US[mask].index
	grid_env_info_US = grid_env_info_US.drop(indices_to_remove)
	print("Shape of grid env info after removing rows with categorical values larger than the number of categories in category {}: ".format(group), grid_env_info_US.shape)
grid_env_info_num = grid_env_info_US.shape[0]

# Include forcing data for the grid env info
# Initialize the forcing data for the grid env info to nan and then fill in the values row by row
forcing_var = ['Input_CWD', 'Input_Litter1', 'Input_Litter2', 
			   'Input_Litter3', 'Altmax_Last_Year', 'Altmax_Current', 
			   'Nbedrock', 'Xio', 'Xin', 'Sand_Content', 'Soil_Temperature', 
			   'Soil_Water']

model_force_pred_input_vector_cwd = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_input_vector_litter1 = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_input_vector_litter2 = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_input_vector_litter3 = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_altmax_lastyear_profile = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_altmax_current_profile = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_nbedrock = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_xio = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_xin = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_sand_vector = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_soil_temp_profile = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_soil_water_profile = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan

# Fill in the forcing data
for irow in np.arange(0, grid_env_info_num):
	lat_loc = np.where(abs(grid_env_info_US.iloc[irow, :]["original_lat"] - lat_grid) == min(abs(grid_env_info_US.iloc[irow, :]["original_lat"] - lat_grid)))[0][0]
	lon_loc = np.where(abs(grid_env_info_US.iloc[irow, :]["original_lon"] - lon_grid) == min(abs(grid_env_info_US.iloc[irow, :]["original_lon"] - lon_grid)))[0][0]
	model_force_pred_input_vector_cwd[irow, :] = cesm2_simu_input_sum_cwd[lat_loc, lon_loc, :]
	model_force_pred_input_vector_litter1[irow, :] = cesm2_simu_input_sum_litter1[lat_loc, lon_loc, :]
	model_force_pred_input_vector_litter2[irow, :] = cesm2_simu_input_sum_litter2[lat_loc, lon_loc, :]
	model_force_pred_input_vector_litter3[irow, :] = cesm2_simu_input_sum_litter3[lat_loc, lon_loc, :]
	model_force_pred_altmax_lastyear_profile[irow, :] = cesm2_simu_altmax_last_year[lat_loc, lon_loc, :]
	model_force_pred_altmax_current_profile[irow, :] = cesm2_simu_altmax[lat_loc, lon_loc, :]
	model_force_pred_nbedrock[irow, :] = cesm2_simu_nbedrock[lat_loc, lon_loc, :]
	model_force_pred_xio[irow, :, :] = cesm2_simu_o_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_xin[irow, :, :] = cesm2_simu_n_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_sand_vector[irow, :, :] = cesm2_simu_cellsand[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_soil_temp_profile[irow, :, :] = cesm2_simu_soil_temperature[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_soil_water_profile[irow, :, :] = cesm2_simu_w_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
# end

# wrapping up for the nn prediction
predict_data_x = np.ones((grid_env_info_num, len(var4nn), 12, 13))*np.nan
predict_data_x[:, 0:len(var4nn), 0, 0] = np.array(grid_env_info_US.loc[:, var4nn])
predict_data_x[:, 0:12, 0, 1] = model_force_pred_input_vector_cwd
predict_data_x[:, 0:12, 0, 2] = model_force_pred_input_vector_litter1
predict_data_x[:, 0:12, 0, 3] = model_force_pred_input_vector_litter2
predict_data_x[:, 0:12, 0, 4] = model_force_pred_input_vector_litter3
predict_data_x[:, 0:12, 0, 5] = model_force_pred_altmax_lastyear_profile
predict_data_x[:, 0:12, 0, 6] = model_force_pred_altmax_current_profile
predict_data_x[:, 0:12, 0, 7] = model_force_pred_nbedrock
predict_data_x[:, 0:20, 0:12, 8] = model_force_pred_xio
predict_data_x[:, 0:20, 0:12, 9] = model_force_pred_xin
predict_data_x[:, 0:20, 0:12, 10] = model_force_pred_sand_vector
predict_data_x[:, 0:20, 0:12, 11] = model_force_pred_soil_temp_profile
predict_data_x[:, 0:20, 0:12, 12] = model_force_pred_soil_water_profile

# create dummy z since it is not used in the prediction
predict_data_z = np.ones((grid_env_info_num))*np.nan
predict_data_c = np.stack([grid_env_info_US["original_lon"], grid_env_info_US["original_lat"]], axis=1)

print("Shape of predict data x", predict_data_x.shape)
print("Shape of predict data z", predict_data_z.shape)
print("Shape of grid env info US", grid_env_info_US.shape)
print(datetime.now(), '------------grid env info prepared------------')

#-----------------------------------------------
# Load PRODA Predicted Parameters for grid data
#-----------------------------------------------
for i in range(1, 10):
	# Loop over all runs (folds) of PRODA. Construct a dataframe where each row
	# is a site. The first column is the profile ID. The next 21 columns are
	# parameters from the first run, next 21 are parameters from the second one, etc.

	# contains one column of profile id
	valid_grid_loc = pd.read_csv(data_dir_input + 'PRODA_Results/valid_grid_loc_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)
	# contains the predicted parameters (21) for each profile
	grid_para = pd.read_csv(data_dir_input + 'PRODA_Results/grid_para_result_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)

	if i == 1:
		# Initialize the dataframe with just profile_id
		grid_PRODA_para = pd.DataFrame(valid_grid_loc)
		grid_PRODA_para.columns = ['profile_id']

		# Concatenate parameters on the right
		grid_PRODA_para = pd.concat([grid_PRODA_para, grid_para], axis = 1)
	else:
		# Concatenate this run's parameters on the right
		grid_PRODA_para = pd.concat([grid_PRODA_para, grid_para], axis = 1)
# end
# For each parameter at each site, take the average across all runs
for i in range(1,22):
	grid_PRODA_para['mean_' + str(i)] = grid_PRODA_para.iloc[:, i:21*10:21].mean(axis = 1)
# end
# Drop the original columns
grid_PRODA_para = grid_PRODA_para.drop(grid_PRODA_para.columns[1:21*9], axis = 1)

# Convert profile ID to zero-based, to match how WOSIS data is processed below
grid_PRODA_para['profile_id'] = grid_PRODA_para['profile_id'] - 1
grid_PRODA_para['profile_id'] = grid_PRODA_para['profile_id'].astype(int)
print("Original grid PRODA para shape", grid_PRODA_para.shape)

# Filter to the 'grid profile IDs' inside the US bounding box
grid_PRODA_para = grid_PRODA_para[grid_PRODA_para['profile_id'].isin(grid_US_profiles)]
print("Grid PRODA para shape after filter to US", grid_PRODA_para.shape)

# First create an empty dataframe with the profile IDs in the same order as grid_env_info_US.
# Then, we attach the PRODA parameters. NOTE: not all profile IDs have PRODA parameters,
# so there may be nans.
grid_PRODA_para_aligned = pd.DataFrame({'profile_id': grid_US_profiles})
grid_PRODA_para_aligned = grid_PRODA_para_aligned.merge(grid_PRODA_para, how='left', on='profile_id')
grid_PRODA_para = grid_PRODA_para_aligned[['mean_1', 'mean_2', 'mean_3', 'mean_4', 'mean_5', 'mean_6', 'mean_7', 'mean_8', 'mean_9', 'mean_10', 'mean_11', \
							                  'mean_12', 'mean_13', 'mean_14', 'mean_15', 'mean_16', 'mean_17', 'mean_18', 'mean_19', 'mean_20', 'mean_21']].to_numpy()
grid_PRODA_para = np.clip(grid_PRODA_para, a_min=0, a_max=1)

#-----------------------------------------------------------------------------------------------
# Old (legacy) model used previously. See mlp_wrapper in mlp.py for a more customizable version.
#------------------------------------------------------------------------------------------------
# define model
class nn_model(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, vectorized, one_hot=False,
				 min_temp=10, max_temp=109, init="xavier_uniform"):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)
		
		This does not include all customization features.
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.vectorized = vectorized

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)

		# Setup categorical encoding
		if self.one_hot:
			# If one-hot, just calculate the input size
			for _, num_classes in self.var_idx_to_emb.items():
				self.new_input_size += num_classes
		else:
			# Create ModuleDict so that all Embedding layers in var_idx_to_emb
			# are registered as parameters
			self.var_idx_to_emb = nn.ModuleDict(self.var_idx_to_emb)
			for _, emb in self.var_idx_to_emb.items():
				self.new_input_size += emb.embedding_dim

		# Number of parameters
		if self.vertical_mixing == 'simple_two_intercepts':
			self.num_params = 22
		else:
			self.num_params = 21

		# Spatial Encoder from PE-GNN
		self.spatial_encoder = GridCellSpatialRelationEncoder(
			spa_embed_dim= 128, 
			coord_dim=2, # Longitude and latitude
			frequency_num=16, 
			max_radius=360,
			min_radius=1e-06,
			freq_init="geometric",
			ffn=True # Enable feedforward network for final spatial embeddings
		)

		# Adjust the input size
		self.new_input_size -= 2  # Remove the longitude and latitude columns
		self.new_input_size += 128  # Add the spatial embeddings

		# Neural network layers
		self.l1 = nn.Linear(self.new_input_size, 128)
		self.l2 = nn.Linear(128, 128)
		self.l4 = nn.Linear(128, 128)
		self.l5 = nn.Linear(128, len(para_index))

		# Dropout layers
		self.dropout = nn.Dropout(0)

		# leaky relu
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)


		# sigmoid parameters
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.min_temp = min_temp
		self.max_temp = max_temp

		# sigmoid
		self.sigmoid = nn.Sigmoid()

		# hardtanh
		# self.hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=True)

		# softsign
		# self.softsign = nn.Softsign()

		# batch normalization
		self.bn1 = nn.BatchNorm1d(128)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn4 = nn.BatchNorm1d(128)

		# Initialize weights
		if init == "xavier_uniform":
			gain_leaky_relu = nn.init.calculate_gain('leaky_relu', 0.3)
			gain_sigmoid = nn.init.calculate_gain('sigmoid')
			nn.init.xavier_uniform_(self.l1.weight, gain=gain_leaky_relu)
			nn.init.xavier_uniform_(self.l2.weight, gain=gain_leaky_relu)
			nn.init.xavier_uniform_(self.l4.weight, gain=gain_leaky_relu)
			nn.init.xavier_uniform_(self.l5.weight, gain=gain_sigmoid)

			# Initialize biases
			nn.init.zeros_(self.l1.bias)
			nn.init.zeros_(self.l2.bias)
			nn.init.zeros_(self.l4.bias)
			nn.init.zeros_(self.l5.bias)
		elif init == 'default':
			pass
		else:
			raise NotImplementedError("Unsupported init")

		# # Transform from 256 to 512
		# self.transform_h1_to_h2 = nn.Linear(256, 512)
		# torch.nn.init.xavier_uniform_(self.transform_h1_to_h2.weight)
		# nn.init.zeros_(self.transform_h1_to_h2.bias)

		# # Transform from 512 to 256
		# self.transform_h3_to_h4 = nn.Linear(512, 256)
		# torch.nn.init.xavier_uniform_(self.transform_h3_to_h4.weight)
		# nn.init.zeros_(self.transform_h3_to_h4.bias)

	def forward(self, input_var, wosis_depth, coords, whether_predict, PRODA_para=None):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth
		coords = coords.unsqueeze(1)  # .detach().cpu().numpy()

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			idx = int(idx)
			if self.one_hot:
				emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				# emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)

		# Spatial Encoding
		spatial_embeddings = self.spatial_encoder(coords)
		spatial_embeddings = spatial_embeddings.squeeze(1) # Remove the channel dimension

		# Remove the longitude and latitude columns from non_categorical_indices
		non_spatial_categorical_indices = [i for i in self.non_categorical_indices if i not in [0, 1]]

		# Concatenate all embeddings
		new_input = torch.concatenate([spatial_embeddings, predictor[:, non_spatial_categorical_indices], all_embs], dim=1)

		# if rank == 0:
		# 	print(f"Spatial embeddings shape: {spatial_embeddings.shape}")
		# 	print(f"Predictor shape: {predictor[:, self.non_categorical_indices].shape}")
		# 	print(f"All embeddings shape: {all_embs.shape}")


		# hidden layers
		h1 = self.l1(new_input)
		h1 = self.bn1(h1)
		h1 = self.leaky_relu(h1)
		h1 = self.dropout(h1)
		# transformed_h1 = self.transform_h1_to_h2(h1)
		h2 = self.l2(h1)
		h2 = self.bn2(h2)
		h2 = self.leaky_relu(h2) # + h1 # residual connection
		h2 = self.dropout(h2)
		# # transformed_h2 = self.transform_h2_to_h3(h2)
		# h3 = self.l3(h2)
		# h3 = self.bn3(h3)
		# h3 = self.leaky_relu(h3) # residual connection
		# h3 = self.dropout(h3)
		# transformed_h3 = self.transform_h3_to_h4(h3)
		h4 = self.l4(h2) ### remember to change back to h2 if only use 4 layers ###
		h4 = self.bn4(h4)
		h4 = self.leaky_relu(h4) # + h3 # residual connection
		h4 = self.dropout(h4)
		# Clamp temp_sigmoid to be between 10 and 200
		# clamped_temp_sigmoid = torch.clamp(self.temp_sigmoid, 10, 200)
		# clamped_temp_sigmoid = 10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Clamp temp_sigmoid to be within a range
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * self.sigmoid(self.temp_sigmoid)  # 10 + 90*self.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# h5 = torch.sigmoid(self.l5(h4) / clamped_temp_sigmoid)
		h5 = self.sigmoid(self.l5(h4)/clamped_temp_sigmoid)

		if PRODA_para is None:   # If PRODA parameters not provided, neural network must output all params
			if self.vertical_mixing == 'simple_two_intercepts':
				assert np.array_equal(para_index, np.arange(22))
			else:
				assert np.array_equal(para_index, np.arange(21))
			PRODA_para = h5
		else:  # Fill the PRODA parameters with predicted parameters if the index of parameters is in the list
			PRODA_para[:, para_index] = h5

		# # check if h5 is nan
		# if torch.isnan(h5).any():
		# 	print("Rank {} h5 is nan".format(os.environ['RANK']))
		# elif torch.isinf(h5).any():
		# 	print("Rank {} h5 is inf".format(os.environ['RANK']))

		######################################################################
		# Attempts at parallelizing the process-based model across examples.
		# Not used currently.
		######################################################################
		# profile_num = h5.shape[0]

		##########################
		# classic python version #
		##########################

		# simu_soc = (torch.ones((profile_num, 200))*np.nan).share_memory_()
		# args_list = [(i, h5[i], forcing[i], obs_depth[i]) for i in range(profile_num)]
		# with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count-4) as executor:
		# 	# Dispatch tasks to worker threads
		# 	for iprofile, profile_output in executor.map(fun_model_simu, args_list):
		# 		simu_soc[iprofile] = profile_output

		# # cleanup after each batch
		# gc.collect()

		#########################
		# torch multiprocessing #
		#########################

		# simu_soc = torch.Tensor(profile_num, 200).share_memory_()
		# simu_soc.fill_(np.nan)

		# # Create a list of arguments to pass to the function
		# args_list = [(i, h5[i], forcing[i], obs_depth[i]) for i in range(profile_num)]

		# with mp.Pool(processes=cpu_count-4) as pool:
		# 	# Dispatch tasks to worker threads
		# 	for iprofile, profile_output in pool.starmap(fun_model_simu, args_list):
		# 			simu_soc[iprofile] = profile_output

		##############################
		# Without Parallel Computing #
		##############################
		if whether_predict == 1:
			simu_soc = fun_model_prediction(PRODA_para, forcing, self.vertical_mixing, self.vectorized)
		else:
			simu_soc = fun_model_simu(PRODA_para, forcing, obs_depth, self.vertical_mixing, self.vectorized)
		return simu_soc, PRODA_para
# end nn_model


# Helper function to combine the training data into a single tensor
class MergeDataset(Dataset):
	def __init__(self, data_x, data_y, data_z, data_c, profile_id, proda_para):
		self.data_x = data_x
		self.data_y = data_y
		self.data_z = data_z
		self.data_c = data_c
		self.profile_id = profile_id
		self.proda_para = proda_para

	def __len__(self):
		return len(self.data_x)

	def __getitem__(self, idx):
		return self.data_x[idx], self.data_y[idx], self.data_z[idx], self.data_c[idx], self.profile_id[idx], self.proda_para[idx]



def create_output_folders(args):
	"""
	Creates and returns a unique job id, based on timestamp, note, PBS job id
	(if exists), seed, cross_val_idx.

	Creates all output folders using this jobID.
	Should only be called once (before spawning processes).

	TODO: This could be instead based on a hash of the hyperparameters.
	This way we would not need to manually specify previous job ID to resume.
	"""

	# @joshuafan: Create a "job id" using timestamp, note, and PBS jobid
	if args.whether_resume == 0:
		# If not resuming, create a new job id
		job_id = time.strftime("%Y%m%d-%H%M%S")  # Convert datetime to string: https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
		if args.note != "":
			job_id += ("_" + args.note)
		pbs_job_id = os.environ.get('PBS_JOBID')
		if pbs_job_id is not None:
			pbs_job_id = pbs_job_id.split('.')[0]
			job_id += ("_" + pbs_job_id)
		job_id += ("_lr={:.0e}".format(args.lr))
		job_id += ("_fold=" + str(args.cross_val_idx))
		job_id += ("_seed=" + str(args.seed))
	else:
		# If resuming, use the same job id as before
		# (Note: if PREVIOUS_JOB_ID environment variable was set, this was already
		# loaded into args.previous_job_id above.)
		job_id = args.previous_job_id

	# Create all necessary output folders
	os.makedirs(os.path.join(data_dir_output, "neural_network"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "model_parameters"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "model_training_history"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, 'neural_network', job_id, 'visualizations'), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "Prediction"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "Bulk_Simulation"), exist_ok=True)
	return job_id


def ddp_setup(rank, world_size):
	"""
	Setup environment parameters and devices.
	Source: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py

	Args:
		rank: Unique identifier of each process
		world_size: Total number of processes
	"""
	# Print number of threads/CPUs
	cpu_count = multiprocessing.cpu_count()
	thread_count = torch.get_num_threads()
	print(datetime.now(), f"========= Setting up DDP. Worker, rank {rank} of {world_size} ==========")
	print("Number of CPUs: ", torch.cpu.device_count())  # No idea why it is not working on NCAR server
	print("Number of Cores: ", cpu_count)
	print("Number of threads: ", thread_count)
	if "CUDA_VISIBLE_DEVICES" in os.environ:
		print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

	# Set number of threads *per worker*. Should equal floor(CPUs/processes)
	torch.set_num_threads(math.floor(torch.get_num_threads() / world_size))

	# Environment variables
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(world_size)
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12356"

	if torch.cuda.is_available():
		# Set device to the appropriate GPU
		gpu = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank])
		device = torch.device(f"cuda:{gpu}")

		# Initialize process group (for GPU, use nccl backend)
		dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=1))
	else:
		# Set device to CPU
		device = torch.device("cpu")

		# Initialize process group (for CPU, use gloo backend)
		dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(hours=1))

	return device


# Start training
def worker(rank, world_size, job_id):

	# Filename to store loss records and visualizations
	nn_training_name = job_id + '_' + model_name
	LOSSES_FILENAME = 'avg_loss_' + nn_training_name + '.csv'
	METRICS_FILENAME = 'avg_metrics_' + nn_training_name + '.csv'
	PLOT_DIR = os.path.join(data_dir_output, 'neural_network', job_id, 'visualizations')
	os.makedirs(PLOT_DIR, exist_ok=True)  # Note: this should already exist from create_output_folders

	# Save printed output to file
	sys.stdout = misc_utils.Logger(os.path.join(data_dir_output, "neural_network", job_id, "output.txt"))

	# Set up distributed environment
	device = ddp_setup(rank, world_size)
	print(f"Finished DDP setup. Rank {rank} of {world_size}. Device {device}. JobID {job_id}.")
	sys.stdout.flush()

	# Create embeddings for categorical variables (each int maps to a different category)
	# If using PyTorch DDP, I think this has to be done inside worker(). Each worker
	# maintains its own copy of the Embedding weights, but they are initialized the same way.
	var_idx_to_emb = dict()  # Column index (before expanding categorical vars) to embedding layer to use
	for group in categorical_vars:
		# Note that within a 'group', variables share embeddings. For example,
		# for 'Texture_USDA_0cm' and 'Texture_USDA_30cm', the embedding of each
		# category is the same.
		n_categories = var_to_categories[group[0]]
		if args.categorical == "embedding":
			emb = nn.Embedding(num_embeddings=n_categories, embedding_dim=args.embed_dim).to(device)
		elif args.categorical == "one_hot":
			emb = n_categories  # Just store the number of categories for one-hot encoding
		else:
			raise ValueError("Invalid value for args.categorical")
		for var in group:
			idx = var4nn.index(var)
			var_idx_to_emb[str(idx)] = emb

	# TODO Not sure if "global model" is correct
	# global model
	if args.model == 'old_mlp':
		model_class = nn_model
		model_kwargs = {"input_vars": len(var4nn),
						"var_idx_to_emb": var_idx_to_emb,
						"vertical_mixing": args.vertical_mixing,
						"vectorized": args.vectorized,
						"one_hot": (args.categorical == "one_hot"),
						"min_temp": args.min_temp,
						"max_temp": args.max_temp,
						"init": args.init}
	elif args.model == 'new_mlp' or args.model == "lipmlp":
		model_class = mlp_wrapper
		model_kwargs = {"input_vars": len(var4nn),
						"var_idx_to_emb": var_idx_to_emb,
						"vertical_mixing": args.vertical_mixing,
						"vectorized": args.vectorized,
						"pos_enc": args.pos_enc,
						"lipschitz": False,
						"one_hot": (args.categorical == "one_hot"),
						"use_bn": args.use_bn,
						"dropout_prob": args.dropout_prob,
						"activation": args.activation,
						"min_temp": args.min_temp,
						"max_temp": args.max_temp,
						"init": args.init,
						"width": args.width,
						"para_index": para_index}
		if args.model == "lipmlp":
			model_kwargs["lipschitz"] = True

	elif args.model == 'nn_only':
		model_class = nn_only

		# Calculate mean/std of Y
		if args.standardize_output:
			output_values = train_y.flatten()[~torch.isnan(train_y.flatten())]
			output_mean = torch.mean(output_values)
			output_std = torch.std(output_values)
		else:
			output_mean, output_std = None, None
		print("NN only, output_mean", output_mean, "output_std", output_std)

		model_kwargs = {"input_vars": len(var4nn),
						"var_idx_to_emb": var_idx_to_emb,
						"pos_enc": args.pos_enc,
						"output_dim": 140,
						"lipschitz": False,
						"one_hot": (args.categorical == "one_hot"),
						"use_bn": args.use_bn,
						"dropout_prob": args.dropout_prob,
						"activation": args.activation,
						"output_mean": output_mean,
						"output_std": output_std,
						"init": args.init,
						"width": args.width}

	else:
		raise ValueError("Invalid args.model")

	# Standardize input if requested
	if args.standardize_input:
		model_kwargs["train_x"] = train_x.to(device)

	if args.whether_resume == 1:
		# Load the model from the checkpoint, and overwrite model_kwargs if saved
		checkpoint_worker = torch.load(checkpoint_path, map_location=device, weights_only=False)
		model_kwargs = checkpoint_worker["model_kwargs"]

	# Create model
	model = model_class(**model_kwargs).to(device)

	# Create distributed version of the model
	if args.use_ddp == 1:
		if torch.cuda.is_available():
			model = DDP(model, device_ids=[device])
		else:  # CPU only 
			model = DDP(model)
		model_without_ddp = model.module
	else:
		model_without_ddp = model

	# Optimizer
	if args.optimizer == "AdamW":
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == "SGD":
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	else:
		raise ValueError("Invalid args.optimizer")

	# If desired: Learning rate scheduler that decays the learning rate throughout training
	if args.scheduler == "reduce_on_plateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)  #, mode="max")
	elif args.scheduler == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
	elif args.scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
	elif args.scheduler == "none":
		scheduler = None
	else:
		raise ValueError("Invalid args.scheduler")

	if args.whether_resume == 1:
		# Load the model from the checkpoint
		state_dict = checkpoint_worker['model_state_dict']
		model.load_state_dict(state_dict)
		optimizer.load_state_dict(checkpoint_worker['optimizer_state_dict'])

	# Loss function
	fun_loss = binns_loss

	# Initialize datasets
	train_dataset = MergeDataset(train_x, train_y, train_z, train_c, train_profile_id, train_proda_para)
	val_dataset = MergeDataset(val_x, val_y, val_z, val_c, val_profile_id, val_proda_para)

	# Use DistributedSampler for distributed training
	if args.use_ddp == 1:
		train_sampler = DistributedSampler(train_dataset)
		val_sampler = DistributedSampler(val_dataset)
	else:
		train_sampler, val_sampler = None, None

	# Data loaders with DistributedSampler
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

	# training and validation loop
	num_epoch = args.n_epochs

	if args.whether_resume == 0:
		# record the loss history
		train_loss_history = np.ones((num_epoch, len(args.losses)))*np.nan
		val_loss_history = np.ones((num_epoch, len(args.losses)))*np.nan
		train_metrics_history = np.ones((num_epoch, 3))*np.nan  # Columns are [MSE, MAE, NSE]
		val_metrics_history = np.ones((num_epoch, 3))*np.nan
		lr_history = np.ones((num_epoch))*np.nan
		best_model_epoch = torch.tensor(0) # epoch with the best model so far

		# Early stopping parameters
		best_val_loss = float('inf') 
		best_val_NSE = float('inf') 
		patience = args.patience
		epochs_without_improvement = 0

		# Save observations
		binn_obs_soc = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		binn_obs_soc[current_data_profile_id, :] = current_data_y

		# Define starting epoch
		start_epoch = 0

		if rank == 1:
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_soc_' + job_id + '.csv', binn_obs_soc, delimiter = ',')
			# print the model structure
			print(model)

		# try to save the predicted parameters before training.
		elif rank == 0:
			val_pred_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, device=device)
			val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, device=device)
			# model.eval()  # TODO Can't really use eval mode before model is trained, since batchnorm stats are not there yet
			with torch.no_grad():
				temp_SOC, temp_pred_para = model(val_x, val_z, val_c, whether_predict=0, PRODA_para=val_proda_para.to(device))
			val_pred_soc[val_profile_id, :] = temp_SOC.detach()
			val_pred_para[val_profile_id, :] = temp_pred_para.detach()
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_training_history/nn_val_pred_soc_' + job_id + "_initial" + '.csv', val_pred_soc.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_parameters/nn_val_pred_soc_' + job_id + "_initial" + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')

	else: 
		# If resuming from a checkpoint, load loss history
		train_loss_history = checkpoint_worker['train_loss_history']
		val_loss_history = checkpoint_worker['val_loss_history']
		train_metrics_history = checkpoint_worker['train_metrics_history']
		val_metrics_history = checkpoint_worker['val_metrics_history']
		lr_history = checkpoint_worker['lr_history']
		best_model_epoch = checkpoint_worker['best_model_epoch']

		# Early stopping parameters
		best_val_loss = checkpoint_worker['best_val_loss']
		best_val_NSE = checkpoint_worker['best_val_NSE'] 
		patience = args.patience
		epochs_without_improvement = checkpoint_worker['epochs_without_improvement']

		# Save observations
		binn_obs_soc = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		binn_obs_soc[current_data_profile_id, :] = current_data_y

		# Define starting epoch
		start_epoch = checkpoint_worker['epoch']

		if rank == 1:
			# print the model structure
			print(model)

	# record start time
	start_time = time.time()
	time_limit_exceeded = False
	whether_break = torch.tensor(0).to(device)

	for iepoch in range(start_epoch, num_epoch):
		epoch_start = time.time()

		# Initialize the break flag for this epoch
		whether_break = torch.tensor(0).to(device)

		# Store predicted para/coords, and predicted/true SOC (for both train and val - for plotting)
		all_train_pred_para = []
		all_train_proda_para = []
		all_train_coords = []
		all_val_pred_para = []
		all_val_proda_para = []
		all_val_coords = []
		all_train_z = []
		all_train_pred_soc = []
		all_train_true_soc = []
		all_val_pred_soc = []
		all_val_true_soc = []
		all_val_z = []

		# clear gradients
		optimizer.zero_grad()
		model.zero_grad()
		if scheduler is not None and rank == 0:
			if iepoch == 0:
				# If scheduler has never been stepped, get_last_lr does not work
				# (https://discuss.pytorch.org/t/how-to-retrieve-learning-rate-from-reducelronplateau-scheduler/54234/3).
				# Instead just read the lr from the commandline arg.
				curr_lr = args.lr
			else:
				print("Output of get_last_lr", scheduler.get_last_lr())
				curr_lr = scheduler.get_last_lr()[0]
			print(f"Epoch {iepoch}: lr = {curr_lr}")
			lr_history[iepoch] = curr_lr
	
		# -------------------------------------training
		loss_record_train = list()  # List of Tensors. Each Tensor contains losses in the order of args.losses.
		metrics_record_train = list()  # List of Tensors (one per batch). Each Tensor contains 3 values: [MSE, MAE, NSE]
		ibatch = 0
		model.train()
		if args.use_ddp == 1:
			train_loader.sampler.set_epoch(iepoch)  # Set sampler's epoch number, so we use a different order per epoch

		# torch.autograd.set_detect_anomaly(True)   # <- helps debug gradient anomalies but is VERY SLOW
		for batch_info in train_loader:
			batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info

			if batch_x.shape[0] == 1 and args.use_bn:  # Batch size of 1 during training does not work with BatchNorm
				continue

			ibatch = ibatch + 1
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			batch_z = batch_z.to(device)
			batch_c = batch_c.to(device)
			batch_proda_para = batch_proda_para.to(device)
			batch_profile_id = batch_profile_id.to(device)

			#------------ 1 forward
			# Model returns predicted (1) SOC, (2) parameters
			batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, PRODA_para=batch_proda_para)

			# Check if batch_pred_para is nan or inf
			if torch.isnan(batch_pred_para).any() or torch.isinf(batch_pred_para).any():
				whether_break = torch.tensor(1).to(device)
				for ipara in range(batch_pred_para.shape[0]):
					if torch.isnan(batch_pred_para[ipara]).any() or torch.isinf(batch_pred_para[ipara]).any():
						print(f"Epoch {iepoch} batch {ibatch} parameter {ipara} is {batch_pred_para[ipara]}")

			# Check for extreme para values
			if args.model != 'nn_only' and (torch.any(batch_pred_para < 0.00001) or torch.any(batch_pred_para > 0.99999)):
				print("Extreme param values")
				print(batch_pred_para)

			#------------ 2 compute the objective function
			l1_loss, smooth_l1_loss, l2_loss, param_reg_loss, train_NSE = fun_loss(batch_y_hat, batch_y, batch_pred_para)

			# Compute additional losses if using. If we are not using them, set them to nan
			lipmlp_loss = np.nan
			spectral_loss = np.nan

			# EXPERIMENTAL: Lipschitz loss if using
			if args.model == "lipmlp" and "lipmlp" in args.losses:
				lipmlp_loss, cs, scalings = model_without_ddp.mlp.get_lipschitz_loss()
				if ibatch == 1:
					print("Lipschitz c", cs, "Scalings", scalings)

			elif "spectral" in args.losses:
				assert args.model == "new_mlp", "Spectral norm regularization only works with --model new_mlp"

				# Compute the spectral norm of the model's layers, and add this as a loss
				spectral_loss = model_without_ddp.mlp.spectral_norm_parallel(device)

			#------------ 3 cleaning gradients
			optimizer.zero_grad()

			#------------ 4 accumulate partical derivatives of objective respect to parameters
			loss_dict = {"l1": l1_loss,
						"smooth_l1": smooth_l1_loss,
						"l2": l2_loss,
						"param_reg": param_reg_loss,
						"lipmlp": lipmlp_loss,
						"spectral": spectral_loss}

			# Store losses in a tensor, in the order of args.losses
			train_losses = torch.stack([loss_dict[loss] for loss in args.losses]).to(device)

			# Compute weighted total loss, backpropagate
			total_loss = torch.dot(train_losses, args.lambdas.to(device))
			total_loss.backward()

			# clip gradients. TODO Not tested.
			if args.clip_value != -1:
				torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)

			#------------ 5 step in the opposite direction of the gradient
			# with torch.no_grad(): para = para - eta*para.grad # eta is learning rate
			optimizer.step()

			# Record losses
			loss_record_train.append(train_losses)
			metrics_record_train.append(torch.tensor([l2_loss.item(), l1_loss.item(), train_NSE.item()], device=device))

			# Record predicted parameters, true/predicted SOC
			all_train_pred_para.append(batch_pred_para)
			all_train_proda_para.append(batch_proda_para)
			all_train_coords.append(batch_c)
			all_train_z.append(batch_z)
			all_train_pred_soc.append(batch_y_hat)
			all_train_true_soc.append(batch_y)

			# flush all printed output
			sys.stdout.flush()
		# end for batch_info in train_loader:

		# Ensure all processes reach this point to synchronize
		# Use all_reduce to check if any process has encountered NaN
		dist.all_reduce(whether_break, op=dist.ReduceOp.MAX)

		# Check if batch_pred_para is nan or inf
		if whether_break.item() == 1:
			print(f"Process {rank} breaking after epoch {iepoch}")
			break  # Break out of the epoch loop if NaN detected in any process

		# training time
		train_time = time.time() - epoch_start

		# Ensure all processes reach this point before proceeding
		dist.barrier()

		# -------------------------------------validation
		loss_record_val = list()  # List of Tensors (one per batch). Each Tensor contains losses in the order of args.losses.
		metrics_record_val = list()  # List of Tensors (one per batch). Each Tensor contains [MSE, MAE, NSE]
		ibatch = 0
		model.eval()
		with torch.no_grad():
			for batch_info in val_loader:
				batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info
				ibatch = ibatch + 1
				batch_x = batch_x.to(device)
				batch_y = batch_y.to(device)
				batch_z = batch_z.to(device)
				batch_c = batch_c.to(device)
				batch_proda_para = batch_proda_para.to(device)
				batch_profile_id = batch_profile_id.to(device)

				# 1 forward
				batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, PRODA_para=batch_proda_para)

				# 2 compute the objective function
				l1_loss, smooth_l1_loss, l2_loss, param_reg_loss, val_NSE = fun_loss(batch_y_hat, batch_y, batch_pred_para)

				# Compute additional losses if using. Not strictly necessary but this helps us see if there
				# is a difference between the losses for train/validation sets
				# If we are not using them, set them to nan
				lipmlp_loss = np.nan
				spectral_loss = np.nan

				if args.model == "lipmlp" and "lipmlp" in args.losses:
					lipmlp_loss, cs, scalings = model_without_ddp.mlp.get_lipschitz_loss()
					if ibatch == 1:
						print("Lipschitz c", cs, "Scalings", scalings)

				elif args.model == "new_mlp" and "spectral" in args.losses:
					# Compute the spectral norm of the model's layers, and add this as a loss
					spectral_loss = model_without_ddp.mlp.spectral_norm_parallel(device)

				loss_dict = {"l1": l1_loss,
				 			"smooth_l1": smooth_l1_loss,
							"l2": l2_loss,
							"param_reg": param_reg_loss,
							"lipmlp": lipmlp_loss,
							"spectral": spectral_loss}

				# Record losses
				# Store losses in a tensor, in the order of args.losses
				val_losses = torch.stack([loss_dict[loss] for loss in args.losses]).to(device)
				loss_record_val.append(val_losses)
				metrics_record_val.append(torch.tensor([l2_loss.item(), l1_loss.item(), val_NSE.item()], device=device))

				# Record predicted parameters, true/predicted SOC
				all_val_pred_para.append(batch_pred_para)
				all_val_proda_para.append(batch_proda_para)
				all_val_coords.append(batch_c)
				all_val_z.append(batch_z)
				all_val_pred_soc.append(batch_y_hat)
				all_val_true_soc.append(batch_y)

				ibatch = ibatch + 1
		# end for batch_info in val_loader: 
			
		# # record the time
		hist_time = time.time() - start_time

		# Gather losses from all processes
		all_train_losses = [torch.zeros((len(args.losses)), device=device) for _ in range(world_size)]
		all_val_losses = [torch.zeros((len(args.losses)), device=device) for _ in range(world_size)]
		all_train_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_train_metrics = [torch.zeros((len(metrics_record_train[0])), device=device) for _ in range(world_size)]
		all_val_metrics = [torch.zeros((len(metrics_record_val[0])), device=device) for _ in range(world_size)]
		all_hist_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]

		dist.all_gather(all_train_losses, torch.stack(loss_record_train, dim=0).mean(dim=0))
		dist.all_gather(all_val_losses, torch.stack(loss_record_val, dim=0).mean(dim=0))
		dist.all_gather(all_train_times, torch.tensor(train_time, device=device))
		dist.all_gather(all_train_metrics, torch.stack(metrics_record_train, dim=0).mean(dim=0))
		dist.all_gather(all_val_metrics, torch.stack(metrics_record_val, dim=0).mean(dim=0))
		dist.all_gather(all_hist_times, torch.tensor(hist_time, device=device))

		# record the loss history
		train_loss_history[iepoch, :] = torch.stack(all_train_losses, dim=0).mean(dim=0).detach().cpu().numpy()
		val_loss_history[iepoch, :] = torch.stack(all_val_losses, dim=0).mean(dim=0).detach().cpu().numpy()
		train_metrics_history[iepoch, :] = torch.stack(all_train_metrics, dim=0).mean(dim=0).detach().cpu().numpy()
		val_metrics_history[iepoch, :] = torch.stack(all_val_metrics, dim=0).mean(dim=0).detach().cpu().numpy()


		####################################################
		## Create true vs predicted scatters per 50 epoch ##
		####################################################
		# To produce comprehensive visualizations, for both train/val sets, create tensors of
		# 1) Predicted parameters for each site
		# 2) PRODA parameters for each site
		# 3) Coordinates (longitude/latitude) of each site
		# 4) Depths for each site/observation (a site may have up to 200 observations, usually much less)
		# 5) Predicted SOC for each site/observation
		# 6) True SOC for each site/observation
		# First aggregate for this rank, then combine all ranks.
		# (Note that DistributedSampler contains repeated examples. We do not remove them.)
		all_train_pred_para = torch.cat(all_train_pred_para, dim=0)  # Pred params for this rank
		all_train_proda_para = torch.cat(all_train_proda_para, dim=0)
		all_train_coords = torch.cat(all_train_coords, dim=0)  # Coords for this rank
		all_train_z = torch.cat(all_train_z, dim=0)
		all_train_pred_soc = torch.cat(all_train_pred_soc, dim=0)  # Pred SOC for this rank
		all_train_true_soc = torch.cat(all_train_true_soc, dim=0)
		all_val_pred_para = torch.cat(all_val_pred_para, dim=0)
		all_val_proda_para = torch.cat(all_val_proda_para, dim=0)
		all_val_coords = torch.cat(all_val_coords, dim=0)
		all_val_z = torch.cat(all_val_z, dim=0)
		all_val_pred_soc = torch.cat(all_val_pred_soc, dim=0)
		all_val_true_soc = torch.cat(all_val_true_soc, dim=0)

		if iepoch % 50 == 0:
			print("Syncing all preds")

			# Estimate max examples per rank. Ok for some to be nan
			train_examples_per_rank = len(train_sampler)  # math.ceil(len(train_sampler) / world_size)
			val_examples_per_rank = len(val_sampler)  # math.ceil(len(val_sampler) / world_size)

			# Pad arrays to this length
			def pad_tensor(tensor, new_length, device):
				"""
				Given tensor of shape [L, D], pads it to shape [new_length, D], where the
				extra rows are filled with nan. new_length must be greater than L.
				"""
				padded = torch.full([new_length, tensor.shape[1]], torch.nan, device=device)
				padded[0:tensor.shape[0]] = tensor
				return padded
			
			all_train_pred_para = pad_tensor(all_train_pred_para, train_examples_per_rank, device)
			all_train_proda_para = pad_tensor(all_train_proda_para, train_examples_per_rank, device)
			all_train_coords = pad_tensor(all_train_coords, train_examples_per_rank, device)
			all_train_z = pad_tensor(all_train_z, train_examples_per_rank, device)
			all_train_pred_soc = pad_tensor(all_train_pred_soc, train_examples_per_rank, device)
			all_train_true_soc = pad_tensor(all_train_true_soc, train_examples_per_rank, device)
			all_val_pred_para = pad_tensor(all_val_pred_para, val_examples_per_rank, device)
			all_val_proda_para = pad_tensor(all_val_proda_para, val_examples_per_rank, device)
			all_val_coords = pad_tensor(all_val_coords, val_examples_per_rank, device)
			all_val_z = pad_tensor(all_val_z, val_examples_per_rank, device)
			all_val_pred_soc = pad_tensor(all_val_pred_soc, val_examples_per_rank, device)
			all_val_true_soc = pad_tensor(all_val_true_soc, val_examples_per_rank, device)

			# Gather SOC/para/coords/depths from all processes
			train_pred_para_list = [torch.full([train_examples_per_rank, model_without_ddp.num_params], torch.nan, device=device) for _ in range(world_size)]  # Empty list of per-rank pred paras
			train_proda_para_list = [torch.full([train_examples_per_rank, model_without_ddp.num_params], torch.nan, device=device) for _ in range(world_size)]  # Empty list of per-rank PRODA paras
			train_coords_list = [torch.full([train_examples_per_rank, 2], torch.nan, device=device) for _ in range(world_size)]
			train_z_list = [torch.full([train_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
			train_pred_soc_list = [torch.full([train_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]  # Empty list of per-rank pred SOCs
			train_true_soc_list = [torch.full([train_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
			val_pred_para_list = [torch.full([val_examples_per_rank, model_without_ddp.num_params], torch.nan, device=device) for _ in range(world_size)]
			val_proda_para_list = [torch.full([val_examples_per_rank, model_without_ddp.num_params], torch.nan, device=device) for _ in range(world_size)]
			val_coords_list = [torch.full([val_examples_per_rank, 2], torch.nan, device=device) for _ in range(world_size)]
			val_z_list = [torch.full([val_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
			val_pred_soc_list = [torch.full([val_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
			val_true_soc_list = [torch.full([val_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
			dist.all_gather(train_pred_para_list, all_train_pred_para)
			dist.all_gather(train_proda_para_list, all_train_proda_para)
			dist.all_gather(train_coords_list, all_train_coords)
			dist.all_gather(train_z_list, all_train_z)
			dist.all_gather(train_pred_soc_list, all_train_pred_soc)
			dist.all_gather(train_true_soc_list, all_train_true_soc)
			dist.all_gather(val_pred_para_list, all_val_pred_para)
			dist.all_gather(val_proda_para_list, all_val_proda_para)
			dist.all_gather(val_coords_list, all_val_coords)
			dist.all_gather(val_z_list, all_val_z)
			dist.all_gather(val_pred_soc_list, all_val_pred_soc)
			dist.all_gather(val_true_soc_list, all_val_true_soc)

			if rank == 0:
				allrank_train_pred_para = torch.cat(train_pred_para_list, dim=0)
				allrank_train_proda_para = torch.cat(train_proda_para_list, dim=0)
				allrank_train_coords = torch.cat(train_coords_list, dim=0)
				allrank_train_z = torch.cat(train_z_list, dim=0)			
				allrank_train_pred_soc = torch.cat(train_pred_soc_list, dim=0)
				allrank_train_true_soc = torch.cat(train_true_soc_list, dim=0)
				allrank_val_pred_para = torch.cat(val_pred_para_list, dim=0)
				allrank_val_proda_para = torch.cat(val_proda_para_list, dim=0)
				allrank_val_coords = torch.cat(val_coords_list, dim=0)
				allrank_val_z = torch.cat(val_z_list, dim=0)
				allrank_val_pred_soc = torch.cat(val_pred_soc_list, dim=0)
				allrank_val_true_soc = torch.cat(val_true_soc_list, dim=0)

				# Scatters of true-vs-predicted SOC (grid).
				# Each row represents a layer (or all layers), each column represents a split (train/val)
				titles = ["Train: All Depths", "Val: All Depths"]
				y_hats = [allrank_train_pred_soc.flatten(), allrank_val_pred_soc.flatten()]  # predictions
				ys = [allrank_train_true_soc.flatten(), allrank_val_true_soc.flatten()]  # labels
				LAYER_BOUNDARIES = [0, 0.1, 0.3, 1.0, 50.0]
				for i in range(len(LAYER_BOUNDARIES) - 1):  # Loop through layers
					layer_loc_train = (allrank_train_z >= LAYER_BOUNDARIES[i]) & (allrank_train_z < LAYER_BOUNDARIES[i+1])  # nan considered false, which is good
					layer_loc_val = (allrank_val_z >= LAYER_BOUNDARIES[i]) & (allrank_val_z < LAYER_BOUNDARIES[i+1])  # nan considered false, which is good
					y_hats.extend([allrank_train_pred_soc[layer_loc_train], allrank_val_pred_soc[layer_loc_val]])
					ys.extend([allrank_train_true_soc[layer_loc_train], allrank_val_true_soc[layer_loc_val]])
					layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
					titles.extend([f'Train: {layer_str}', f'Val: {layer_str}'])
				visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"epoch{iepoch}_scatters.png"), y_hats, ys, titles, cols=2)

				# Maps of true-vs-predicted SOC (grid)
				# Each row represents a layer, each column represents a split (train/val) and {true or predicted}
				lons_list = []
				lats_list = []
				values_list = []
				vars_list = []
				for i in range(len(LAYER_BOUNDARIES) - 1):
					# For each site: compute average SOC over observations in this layer
					layer_loc_train = (allrank_train_z >= LAYER_BOUNDARIES[i]) & (allrank_train_z < LAYER_BOUNDARIES[i+1])  # nan considered false, which is good
					train_true_soc = torch.where(layer_loc_train, allrank_train_true_soc, torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
					train_true_soc = torch.nanmean(train_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
					train_pred_soc = torch.where(layer_loc_train, allrank_train_pred_soc, torch.nan)  # Same for predictions
					train_pred_soc = torch.nanmean(train_pred_soc, dim=1)

					# Repeat above for val data
					layer_loc_val = (allrank_val_z >= LAYER_BOUNDARIES[i]) & (allrank_val_z < LAYER_BOUNDARIES[i+1])  # nan considered false, which is good
					val_true_soc = torch.where(layer_loc_val, allrank_val_true_soc, torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
					val_true_soc = torch.nanmean(val_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
					val_pred_soc = torch.where(layer_loc_val, allrank_val_pred_soc, torch.nan)  # Same for predictions
					val_pred_soc = torch.nanmean(val_pred_soc, dim=1)

					# Collect results
					lons_list.extend([allrank_train_coords[:, 0], allrank_train_coords[:, 0], allrank_val_coords[:, 0], allrank_val_coords[:, 0]])
					lats_list.extend([allrank_train_coords[:, 1], allrank_train_coords[:, 1], allrank_val_coords[:, 1], allrank_val_coords[:, 1]])
					values_list.extend([train_true_soc, train_pred_soc, val_true_soc, val_pred_soc])
					layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
					vars_list.extend([f'True SOC - Train: {layer_str}', f'Predicted SOC - Train: {layer_str}',
					   				  f'True SOC - Val: {layer_str}', f'Predicted SOC - Val: {layer_str}'])
				visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"epoch{iepoch}_soc_maps.png"),
						lons_list, lats_list, values_list, vars_list, us_only=True, cols=4)

				# Parameter maps. Each row is a parameter, each column represents a split (train/val)
				if args.model != "nn_only":
					if args.synthetic_labels:
						# If synthetic labels, we also have labels for parameters, so we can compare predicted vs true
						lons_list = []
						lats_list = []
						values_list = []
						vars_list = []
						for para_idx in range(model_without_ddp.num_params):
							lons_list.extend([allrank_train_coords[:, 0], allrank_train_coords[:, 0], allrank_val_coords[:, 0], allrank_val_coords[:, 0]])
							lats_list.extend([allrank_train_coords[:, 1], allrank_train_coords[:, 1], allrank_val_coords[:, 1], allrank_val_coords[:, 1]])
							values_list.extend([allrank_train_proda_para[:, para_idx], allrank_train_pred_para[:, para_idx],
						   						allrank_val_proda_para[:, para_idx], allrank_val_pred_para[:, para_idx]])
							para_name = para_names[para_idx]
							vars_list.extend([f'True para {para_name} - Train', f'Predicted para {para_name} - Train',
											  f'True para {para_name} - Val', f'Predicted para {para_name} - Val'])
						visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"epoch{iepoch}_para_maps.png"),
								lons_list, lats_list, values_list, vars_list, us_only=True, cols=4)

						# Also plot scatters
						y_hats = []
						ys = []
						titles = []
						for para_idx in range(model_without_ddp.num_params):
							y_hats.extend([allrank_train_pred_para[:, para_idx], allrank_val_pred_para[:, para_idx]])
							ys.extend([allrank_train_proda_para[:, para_idx], allrank_val_proda_para[:, para_idx]])
							para_name = para_names[para_idx]
							titles.extend([f'Train: {para_name}', f'Val: {para_name}'])
						visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"epoch{iepoch}_para_scatters.png"), y_hats, ys, titles, cols=2)

					else:
						# If using real labels, we do not have labels for parameters, so only plot the predictions
						lons_list = []
						lats_list = []
						values_list = []
						vars_list = []
						for para_idx in range(model_without_ddp.num_params):
							lons_list.extend([allrank_train_coords[:, 0], allrank_val_coords[:, 0]])
							lats_list.extend([allrank_train_coords[:, 1], allrank_val_coords[:, 1]])
							values_list.extend([allrank_train_pred_para[:, para_idx], allrank_val_pred_para[:, para_idx]])
							vars_list.extend([f'Train: {para_names[para_idx]}', f'Val: {para_names[para_idx]}'])
						visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"epoch{iepoch}_para_maps.png"),
								lons_list, lats_list, values_list, vars_list, us_only=True, cols=2)


		if rank == 0:
			# Save loss history
			train_losses_epoch = {loss: round(train_loss_history[iepoch, loss_idx], 2) for loss_idx, loss in enumerate(args.losses)}
			val_losses_epoch = {loss: round(val_loss_history[iepoch, loss_idx], 2) for loss_idx, loss in enumerate(args.losses)}
			train_NSE = round(train_metrics_history[iepoch, 2], 2)  # NSE is column 2
			val_NSE = round(val_metrics_history[iepoch, 2], 2)
			print(f'Epoch {iepoch} Rank {rank} - Train NSE: {train_NSE}, validation NSE: {val_NSE}, time: {train_time:.2f}', flush=True)
			print(f'Train losses ({all_train_pred_soc.shape[0]} examples): {train_losses_epoch}')
			print(f'Validation losses ({all_val_pred_soc.shape[0]} examples): {val_losses_epoch}')

			# If this model is the best so far, save the checkpoint into 'opt_nn_{job_id}.pt'
			if val_metrics_history[iepoch, 2] <= best_val_NSE:  # @joshuafan: removed the iepoch==0 condition
				print(f'Best model updated at epoch {iepoch}')
				best_model_epoch = torch.tensor(iepoch, device=device)

				checkpoint_best_model = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'model_kwargs': model_kwargs,  # Save kwargs used to construct the model
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss': best_val_loss,
					'best_val_NSE': best_val_NSE,
					'best_model_epoch': best_model_epoch,
					'train_loss_history': train_loss_history,
					'val_loss_history': val_loss_history,
					'train_metrics_history': train_metrics_history,
					'val_metrics_history': val_metrics_history,
					'lr_history': lr_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
				}
				
				best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
				torch.save(checkpoint_best_model, best_model_path)
			# end if val_metrics_history[iepoch, 2] <= best_val_NSE:

			# save the training and validation loss history
			train_time = torch.stack(all_train_times).mean().item()
			hist_time = torch.stack(all_hist_times).mean().item()
			loss_file = os.path.join(data_dir_output, "neural_network", job_id, LOSSES_FILENAME)
			if iepoch == 0:  # Write the header if the file doesn't exist yet
				with open(loss_file, mode='w') as f:
					csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['epoch'] + [f'{loss}_loss_train' for loss in args.losses] +
										[f'{loss}_loss_val' for loss in args.losses] +
										['epoch_time', 'cumulative_time', 'best_model_epoch'])
			with open(loss_file, mode='a+') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow([iepoch] + [train_loss_history[iepoch, loss_idx] for loss_idx, loss in enumerate(args.losses)] +
									[val_loss_history[iepoch, loss_idx] for loss_idx, loss in enumerate(args.losses)] +
									[round(train_time, 2), round(hist_time, 2), best_model_epoch.item()])

			# NSE file. TODO Also make this into a CSV.
			nse_file = os.path.join(data_dir_output, "neural_network", job_id, METRICS_FILENAME)
			if iepoch == 0:
				with open(nse_file, mode='w') as f:
					csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['epoch', 'train_MSE', 'train_MAE', 'train_NSE', 'val_MSE', 'val_MAE', 'val_NSE', 'epoch_time', 'cumulative_time', 'best_model_epoch'])
			with open(nse_file, mode='a+') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow([iepoch] + torch.stack(all_train_metrics, dim=0).mean(dim=0).tolist() +
									 torch.stack(all_val_metrics, dim=0).mean(dim=0).tolist() +
									 [round(train_time, 2) + round(hist_time, 2) + best_model_epoch.item()])

		# Ensure all processes reach this point before proceeding
		dist.barrier()

		if args.scheduler == "reduce_on_plateau":
			scheduler.step(val_metrics_history[iepoch, 2])
		elif scheduler is not None:
			scheduler.step()
		if rank == 0 and scheduler is not None:
			print("New learning rate =", scheduler.get_last_lr())

		# Add a early stopping condition
		if val_metrics_history[iepoch, 2] < best_val_NSE:
			best_val_NSE = val_metrics_history[iepoch, 2]
			best_val_loss = val_loss_history[iepoch, :]
			epochs_without_improvement = 0
			# Optionally save the model here if it's the best one so far
		else:
			epochs_without_improvement += 1

		# Early stopping condition
		if epochs_without_improvement == patience:
			print("Rank {}: Early stopping due to no improvement after {} epochs.".format(rank, patience))
			break  # exit the epoch loop
		
		# Save checkpoint (1) every save_freq epochs or (2) time limit exceeded 
		time_limit_exceeded = (time.time() - job_begin_time > args.time_limit * 3600)
		if time_limit_exceeded or iepoch % args.save_freq == 0:
			if rank == 0:
				checkpoint = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'model_kwargs': model_kwargs,  # Save kwargs used to construct the model
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss': best_val_loss,
					'best_val_NSE': best_val_NSE,
					'best_model_epoch': best_model_epoch,
					'train_loss_history': train_loss_history,
					'val_loss_history': val_loss_history,
					'train_metrics_history': train_metrics_history,
					'val_metrics_history': val_metrics_history,
					'lr_history': lr_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
				}
				torch.save(checkpoint, data_dir_output + 'neural_network/' + job_id + '/checkpoint_' + job_id + '.pt')

				if time_limit_exceeded:
					print("Rank {}: Runtime exceeded, saving checkpoint and exiting.".format(rank))

					# Create a file to submit the job again
					if args.job_scheduler == 'slurm':
						# Create a file to submit the job again
						with open(job_submit_path + 'Resume' + job_id + '.submit', 'w') as f:
							f.write(f'#!/bin/bash\n')
							f.write(f'#SBATCH -p full\n')
							f.write(f'#SBATCH -J binn_resume\n')
							f.write(f'#SBATCH --gpus {args.num_CPU}\n')
							f.write(f'#SBATCH -c {args.num_CPU*2}\n')
							f.write(f'#SBATCH -N 1 -n 1\n')
							f.write(f'#SBATCH --mem=50GB\n')
							f.write(f'#SBATCH -t 12:00:00\n')
							f.write(f'source ~/.bashrc\n')
							f.write(f'module load cuda\n')
							f.write(f'conda activate binn\n\n')
							f.write(f'python {" ".join(sys.argv)} --time_limit 11.5 --whether_resume 1\n')

						# submit the job again
						submit_command = ['sbatch',
							f'--export=PREVIOUS_JOB_ID={job_id}',
							job_submit_path + 'Resume' + job_id + '.submit']
						# Submit the job and get the new job ID
						try:
							submit_output = subprocess.check_output(submit_command, universal_newlines=True)
							new_job_id = submit_output.strip()
							print(f"New job submitted. New Job ID is {new_job_id}")
						except subprocess.CalledProcessError as e:
							print(f"Failed to submit job: {e.output}")
					else:
						with open(job_submit_path + 'Resume' + job_id + '.submit', 'w') as f:
							f.write(f'#!/bin/bash\n')
							f.write(f'#PBS -A UOKL0017\n')
							f.write(f'#PBS -N DDP_BINN_Resume\n')
							f.write(f'#PBS -q main\n')
							f.write(f'#PBS -l walltime=12:00:00\n')
							f.write(f'#PBS -l select=1:ncpus=128\n\n')
							f.write(f'# Use scratch for temporary files to avoid space limits in /tmp\n')
							f.write(f'export TMPDIR=/glade/scratch/$USER/temp\n')
							f.write(f'mkdir -p $TMPDIR\n\n')
							f.write(f'# Load modules to match compile-time environment\n')
							f.write(f'module purge\n')
							f.write(f'module load conda\n')
							f.write(f'module load cuda\n\n')
							f.write(f'# Activate environment in conda\n')
							f.write(f'conda activate BINN_310_CPU\n\n')
							f.write(f'# Start the Python Code\n')
							f.write(f'python -u /glade/u/home/haodixu/BINN/Server_Script/binns_DDP.py {" ".join(sys.argv)} --whether_resume 1\n')
							# f.write(f'python -u /glade/u/home/haodixu/BINN/Server_Script/binns_DDP.py --lr ' + str(args.lr) + ' --weight_decay ' + str(args.weight_decay) + ' --batch_size ' + str(args.batch_size) + \
							# 	' --seed ' + str(args.seed) + ' --n_epochs ' + str(args.n_epochs) + ' --patience ' + str(args.patience) + ' --model ' + str(args.model) + ' --lambda_lipschitz ' + str(args.lambda_lipschitz) + \
							# 	' --note ' + str(args.note) + ' --categorical ' + str(args.categorical) + ' --use_bn ' + ' --embed_dim ' + str(args.embed_dim) + ' --cross_val_idx' + str(args.cross_val_idx) + \
							# 	' --num_CPU ' + str(args.num_CPU) + ' --whether_resume 1\n')

						# submit the job again
						submit_command = ['qsub', 
							'-v', f"PREVIOUS_JOB_ID={job_id}",
							job_submit_path + 'Resume' + job_id + '.submit']
						# Submit the job and get the new job ID
						try:
							submit_output = subprocess.check_output(submit_command, universal_newlines=True)
							new_job_id = submit_output.strip()
							print(f"New job submitted. New Job ID is {new_job_id}")
						except subprocess.CalledProcessError as e:
							print(f"Failed to submit job: {e.output}")
					break

	print(f"Rank {rank} finished processing data.")

	if time_limit_exceeded:
		print(f"Rank {rank}: Exiting after saving checkpoint.")
		return
	if whether_break.item() == 1:
		print(f"Rank {rank}: Exiting after training due to NaN encountered in any process.")
		return

	# Ensure all processes reach the end
	dist.barrier()


	##################################################
	# Done training. Load best model for analysis
	##################################################
	# TODO: weights_only=False is not recommended. Should modify code to only save tensors.
	new_checkpoint = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt', map_location=device, weights_only=False)
	best_guess_model = model  # Do not need to create a new model
	best_guess_model.load_state_dict(new_checkpoint['model_state_dict'])
	print("Loaded model for rank: {}".format(rank))

	dist.barrier()

	if rank == 0:
		# Plot loss curves throughout training. Normalize each curve relative to its mean,
		# to make the scales comparable.
		train_loss_history = train_loss_history[~np.any(np.isnan(train_loss_history), axis=1)]
		val_loss_history = val_loss_history[~np.any(np.isnan(val_loss_history), axis=1)]
		losses = [(train_loss_history[:, loss_idx] / train_loss_history[:, loss_idx].mean()) for loss_idx in range(len(args.losses))] + \
				[(val_loss_history[:, loss_idx] / val_loss_history[:, loss_idx].mean()) for loss_idx in range(len(args.losses))]
		labels = [f"{loss} loss (train)" for loss in args.losses] + [f"{loss} loss (val)" for loss in args.losses]
		visualization_utils.plot_losses(os.path.join(PLOT_DIR, "losses.png"), losses, labels)

		# Also plot NSE curves: first remove nans. Note NSE is column 2.
		train_NSE_list = train_metrics_history[~np.any(np.isnan(train_metrics_history), axis=1), 2].flatten().tolist()
		val_NSE_list = val_metrics_history[~np.any(np.isnan(val_metrics_history), axis=1), 2].flatten().tolist()
		visualization_utils.plot_losses(os.path.join(PLOT_DIR, "nses.png"),
									    [train_NSE_list, val_NSE_list],
										["Train NSE", "Val NSE"],
										min_val=0, max_val=1.2)

		# Plot learning rate schedule
		if scheduler is not None:
			plt.plot(np.arange(lr_history.size), lr_history)
			plt.xlabel('Epoch #')
			plt.ylabel('Learning rate')
			plt.title('Learning rate schedule')
			plt.savefig(os.path.join(PLOT_DIR, "lr_schedule.png"))
			plt.close()

		#######################################################
		# Get best model's predictions on train/val/test sets
		#######################################################
		print("Rank 0 beginning prediction at time {}".format(datetime.now()))
		best_guess_model.eval()
		print("Rank 0 model set to eval at time {}".format(datetime.now()))
		with torch.no_grad():
			# Get predictions for train examples, compute loss & plot
			best_guess_train_y_hat, best_guess_train_pred_para = best_guess_model(train_x.to(device), train_z.to(device), train_c.to(device),
																			      whether_predict=0, PRODA_para=train_proda_para.to(device))
			train_mae, train_smooth_l1_loss, train_mse, _, train_NSE = fun_loss(best_guess_train_y_hat, train_y.to(device), best_guess_train_pred_para)
			print(f'Train - MSE: {train_mse.item():.2f}, MAE: {train_mae.item():.2f}, NSE: {train_NSE.item():.2f}')

			# Get predictions for val examples, compute loss & plot
			best_guess_val_y_hat, best_guess_val_pred_para = best_guess_model(val_x.to(device), val_z.to(device), val_c.to(device),
																			  whether_predict=0, PRODA_para=val_proda_para.to(device))
			val_mae, val_smooth_l1_loss, val_mse, _, val_NSE = fun_loss(best_guess_val_y_hat, val_y.to(device), best_guess_val_pred_para)
			print(f'Val - MSE: {val_mse.item():.2f}, MAE: {val_mae.item():.2f}, NSE: {val_NSE.item():.2f}')

			if test_split_ratio != 0:
				# Get predictions for test examples, compute loss & plot
				best_guess_test_y_hat, best_guess_test_pred_para = best_guess_model(test_x.to(device), test_z.to(device), test_c.to(device),
																				    whether_predict=0, PRODA_para=test_proda_para.to(device))
				test_mae, test_smooth_l1_loss, test_mse, _, test_NSE = fun_loss(best_guess_test_y_hat, test_y.to(device), best_guess_test_pred_para)
				print(f'Test - MSE: {test_mse.item():.2f}, MAE: {test_mae.item():.2f}, NSE: {test_NSE.item():.2f}')

		# Summary csv file of all results. Create this if it doesn't exist
		results_summary_file = os.path.join(data_dir_output, f"neural_network/results_summary_{args.note}.csv")
		if not os.path.isfile(results_summary_file):
			with open(results_summary_file, mode='w') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow(['job_id', 'command', 'data_string', 'lr', 'weight_decay', 'seed', 'model_path', 'val_MSE', 'val_MAE', 'val_NSE', 'test_MSE', 'test_MAE', 'test_NSE'])
		command_string = " ".join(sys.argv)
		data_string = f"Fold {args.cross_val_idx} {args.split} (data_seed = {args.data_seed}, n_datapoints = {args.n_datapoints})"

		# Add a row to the summary csv file
		with open(results_summary_file, mode='a+') as f:
			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
			csv_writer.writerow([job_id, command_string, data_string, args.lr, args.weight_decay, args.seed, best_model_path, val_mse.item(), val_mae.item(), val_NSE.item(), test_mse.item(), test_mae.item(), test_NSE.item()])


		# create folder for the results
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Validation', exist_ok=True)
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Train', exist_ok=True)
		if test_split_ratio != 0:
			os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Test', exist_ok=True)

		#############
		# Test Data #
		#############

		## predictions and parameters for the test profiles ##

		best_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, device=device)  # dtype = torch.float32, 
		best_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, device=device)  # dtype = torch.float32, 
		upper_depth_all = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		lower_depth_all = np.ones((wosis_profile_info.shape[0], 200))*np.nan

		
		upper_depth_all[current_data_profile_id, :] = obs_upper_depth_matrix
		lower_depth_all[current_data_profile_id, :] = obs_lower_depth_matrix

		best_simu_soc[test_profile_id, :] = best_guess_test_y_hat
		best_pred_para[test_profile_id, :] = best_guess_test_pred_para


		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_soc_' + job_id + '.csv', best_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_pred_para_' + job_id + '.csv', best_pred_para.detach().cpu().numpy(), delimiter = ',')

		## bulk convergence ##
		# initializz a seperate array to store the prediction results for the test profiles
		# with return of the function: carbon_input, cpool_steady_state, cpools_layer, soc_layer, total_res_time, total_res_time_base, res_time_base_pools, t_scaler, bulk_A, w_scaler, bulk_K, bulk_V, bulk_xi, bulk_I, litter_fraction
		if args.model != 'nn_only': 
			carbon_input_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			cpool_steady_state_test = np.ones((wosis_profile_info.shape[0], 140))*np.nan
			cpools_layer_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			soc_layer_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			total_res_time_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			total_res_time_base_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			res_time_base_pools_test = np.ones((wosis_profile_info.shape[0], 140))*np.nan
			t_scaler_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			bulk_A_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			w_scaler_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			bulk_K_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_V_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_xi_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_I_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			litter_fraction_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan

			carbon_input_test_profile, cpool_steady_state_test_profile, cpools_layer_test_profile, \
				soc_layer_test_profile, total_res_time_test_profile, total_res_time_base_test_profile, res_time_base_pools_test_profile, \
					t_scaler_test_profile, bulk_A_test_profile, w_scaler_test_profile, bulk_K_test_profile, bulk_V_test_profile, bulk_xi_test_profile, \
						bulk_I_test_profile, litter_fraction_test_profile = fun_bulk_simu(best_guess_test_pred_para.to(device), test_x.to(device), args.vertical_mixing, args.vectorized)
			
			# store the results
			carbon_input_test[test_profile_id, :] = carbon_input_test_profile.detach().cpu().numpy()
			cpool_steady_state_test[test_profile_id, :] = cpool_steady_state_test_profile.detach().cpu().numpy()
			cpools_layer_test[test_profile_id, :] = cpools_layer_test_profile.detach().cpu().numpy()
			soc_layer_test[test_profile_id, :] = soc_layer_test_profile.detach().cpu().numpy()
			total_res_time_test[test_profile_id, :] = total_res_time_test_profile.detach().cpu().numpy()
			total_res_time_base_test[test_profile_id, :] = total_res_time_base_test_profile.detach().cpu().numpy()
			res_time_base_pools_test[test_profile_id, :] = res_time_base_pools_test_profile.detach().cpu().numpy()
			t_scaler_test[test_profile_id, :] = t_scaler_test_profile.detach().cpu().numpy()
			bulk_A_test[test_profile_id, :] = bulk_A_test_profile.detach().cpu().numpy()
			w_scaler_test[test_profile_id, :] = w_scaler_test_profile.detach().cpu().numpy()
			bulk_K_test[test_profile_id, :] = bulk_K_test_profile.detach().cpu().numpy()
			bulk_V_test[test_profile_id, :] = bulk_V_test_profile.detach().cpu().numpy()
			bulk_xi_test[test_profile_id, :] = bulk_xi_test_profile.detach().cpu().numpy()
			bulk_I_test[test_profile_id, :] = bulk_I_test_profile.detach().cpu().numpy()
			litter_fraction_test[test_profile_id, :] = litter_fraction_test_profile.detach().cpu().numpy()

			# save data
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_carbon_input_' + job_id + '.csv', carbon_input_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_soc_layer_' + job_id + '.csv', soc_layer_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_total_res_time_' + job_id + '.csv', total_res_time_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_t_scaler_' + job_id + '.csv', t_scaler_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_A_' + job_id + '.csv', bulk_A_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_w_scaler_' + job_id + '.csv', w_scaler_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_K_' + job_id + '.csv', bulk_K_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_V_' + job_id + '.csv', bulk_V_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_I_' + job_id + '.csv', bulk_I_test, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_test, delimiter = ',')



		############
		# Val Data #
		############

		## predictions and parameters for the validation profiles ##

		# initializz a seperate array to store the prediction results for the validation profiles
		val_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, device=device)  # dtype = torch.float32,
		val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, device=device)  # dtype = torch.float32,

		val_simu_soc[val_profile_id, :] = best_guess_val_y_hat
		val_pred_para[val_profile_id, :] = best_guess_val_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_soc_' + job_id + '.csv', val_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_pred_para_' + job_id + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')

		## bulk convergence ##
		if args.model != 'nn_only': 
			carbon_input_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			cpool_steady_state_val = np.ones((wosis_profile_info.shape[0], 140))*np.nan
			cpools_layer_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			soc_layer_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			total_res_time_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			total_res_time_base_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			res_time_base_pools_val = np.ones((wosis_profile_info.shape[0], 140))*np.nan
			t_scaler_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			bulk_A_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			w_scaler_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			bulk_K_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_V_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_xi_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_I_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			litter_fraction_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan

			carbon_input_val_profile, cpool_steady_state_val_profile, cpools_layer_val_profile, \
				soc_layer_val_profile, total_res_time_val_profile, total_res_time_base_val_profile, res_time_base_pools_val_profile, \
					t_scaler_val_profile, bulk_A_val_profile, w_scaler_val_profile, bulk_K_val_profile, bulk_V_val_profile, bulk_xi_val_profile, \
						bulk_I_val_profile, litter_fraction_val_profile = fun_bulk_simu(best_guess_val_pred_para.to(device), val_x.to(device), args.vertical_mixing, args.vectorized)
			
			# store the results
			carbon_input_val[val_profile_id, :] = carbon_input_val_profile.detach().cpu().numpy()
			cpool_steady_state_val[val_profile_id, :] = cpool_steady_state_val_profile.detach().cpu().numpy()
			cpools_layer_val[val_profile_id, :] = cpools_layer_val_profile.detach().cpu().numpy()
			soc_layer_val[val_profile_id, :] = soc_layer_val_profile.detach().cpu().numpy()
			total_res_time_val[val_profile_id, :] = total_res_time_val_profile.detach().cpu().numpy()
			total_res_time_base_val[val_profile_id, :] = total_res_time_base_val_profile.detach().cpu().numpy()
			res_time_base_pools_val[val_profile_id, :] = res_time_base_pools_val_profile.detach().cpu().numpy()
			t_scaler_val[val_profile_id, :] = t_scaler_val_profile.detach().cpu().numpy()
			bulk_A_val[val_profile_id, :] = bulk_A_val_profile.detach().cpu().numpy()
			w_scaler_val[val_profile_id, :] = w_scaler_val_profile.detach().cpu().numpy()
			bulk_K_val[val_profile_id, :] = bulk_K_val_profile.detach().cpu().numpy()
			bulk_V_val[val_profile_id, :] = bulk_V_val_profile.detach().cpu().numpy()
			bulk_xi_val[val_profile_id, :] = bulk_xi_val_profile.detach().cpu().numpy()
			bulk_I_val[val_profile_id, :] = bulk_I_val_profile.detach().cpu().numpy()
			litter_fraction_val[val_profile_id, :] = litter_fraction_val_profile.detach().cpu().numpy()

			# save data
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_carbon_input_' + job_id + '.csv', carbon_input_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_soc_layer_' + job_id + '.csv', soc_layer_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_total_res_time_' + job_id + '.csv', total_res_time_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_t_scaler_' + job_id + '.csv', t_scaler_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_A_' + job_id + '.csv', bulk_A_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_w_scaler_' + job_id + '.csv', w_scaler_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_K_' + job_id + '.csv', bulk_K_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_V_' + job_id + '.csv', bulk_V_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_I_' + job_id + '.csv', bulk_I_val, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_val, delimiter = ',')

		##############
		# Train Data #
		##############

		## predictions and parameters for the training profiles ##

		# initializz a seperate array to store the prediction results for the training profiles
		train_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, device=device)  # dtype = torch.float32, 
		train_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, device=device)  # dtype = torch.float32, 

		train_simu_soc[train_profile_id, :] = best_guess_train_y_hat
		train_pred_para[train_profile_id, :] = best_guess_train_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_soc_' + job_id + '.csv', train_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_pred_para_' + job_id + '.csv', train_pred_para.detach().cpu().numpy(), delimiter = ',')

		## bulk convergence ##
		if args.model != 'nn_only': 
			carbon_input_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			cpool_steady_state_train = np.ones((wosis_profile_info.shape[0], 140))*np.nan
			cpools_layer_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			soc_layer_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			total_res_time_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			total_res_time_base_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			res_time_base_pools_train = np.ones((wosis_profile_info.shape[0], 140))*np.nan
			t_scaler_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			bulk_A_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			w_scaler_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
			bulk_K_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_V_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_xi_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			bulk_I_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
			litter_fraction_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan

			carbon_input_train_profile, cpool_steady_state_train_profile, cpools_layer_train_profile, \
				soc_layer_train_profile, total_res_time_train_profile, total_res_time_base_train_profile, res_time_base_pools_train_profile, \
					t_scaler_train_profile, bulk_A_train_profile, w_scaler_train_profile, bulk_K_train_profile, bulk_V_train_profile, bulk_xi_train_profile, \
						bulk_I_train_profile, litter_fraction_train_profile = fun_bulk_simu(best_guess_train_pred_para.to(device), train_x.to(device), args.vertical_mixing, args.vectorized)
			
			# store the results
			carbon_input_train[train_profile_id, :] = carbon_input_train_profile.detach().cpu().numpy()
			cpool_steady_state_train[train_profile_id, :] = cpool_steady_state_train_profile.detach().cpu().numpy()
			cpools_layer_train[train_profile_id, :] = cpools_layer_train_profile.detach().cpu().numpy()
			soc_layer_train[train_profile_id, :] = soc_layer_train_profile.detach().cpu().numpy()
			total_res_time_train[train_profile_id, :] = total_res_time_train_profile.detach().cpu().numpy()
			total_res_time_base_train[train_profile_id, :] = total_res_time_base_train_profile.detach().cpu().numpy()
			res_time_base_pools_train[train_profile_id, :] = res_time_base_pools_train_profile.detach().cpu().numpy()
			t_scaler_train[train_profile_id, :] = t_scaler_train_profile.detach().cpu().numpy()
			bulk_A_train[train_profile_id, :] = bulk_A_train_profile.detach().cpu().numpy()
			w_scaler_train[train_profile_id, :] = w_scaler_train_profile.detach().cpu().numpy()
			bulk_K_train[train_profile_id, :] = bulk_K_train_profile.detach().cpu().numpy()
			bulk_V_train[train_profile_id, :] = bulk_V_train_profile.detach().cpu().numpy()
			bulk_xi_train[train_profile_id, :] = bulk_xi_train_profile.detach().cpu().numpy()
			bulk_I_train[train_profile_id, :] = bulk_I_train_profile.detach().cpu().numpy()
			litter_fraction_train[train_profile_id, :] = litter_fraction_train_profile.detach().cpu().numpy()

			# save data
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_carbon_input_' + job_id + '.csv', carbon_input_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_soc_layer_' + job_id + '.csv', soc_layer_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_total_res_time_' + job_id + '.csv', total_res_time_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_t_scaler_' + job_id + '.csv', t_scaler_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_A_' + job_id + '.csv', bulk_A_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_w_scaler_' + job_id + '.csv', w_scaler_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_K_' + job_id + '.csv', bulk_K_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_V_' + job_id + '.csv', bulk_V_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_I_' + job_id + '.csv', bulk_I_train, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_train, delimiter = ',')


		#############
		# Test Maps #
		#############

		# get the latitudes and longitudes of the test profiles by matching ProfileID in env_info with the test_profile_id
		test_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
		test_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
		test_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
		test_profile_id_all[test_profile_id] = test_profile_id
		test_profile_id_num = test_profile_id.numpy().astype(int)
		# check shape
		print("test_profile_id_num.shape: ", test_profile_id_num.shape)
		# print the range of test_profile_id_num
		print("test_profile_id_num.min(): ", test_profile_id_num.min())
		print("test_profile_id_num.max(): ", test_profile_id_num.max())
		print("env_info.shape: ", env_info.shape)
		test_lons[test_profile_id_num] = np.array(env_info.loc[test_profile_id_num, "original_lon"])
		test_lats[test_profile_id_num] = np.array(env_info.loc[test_profile_id_num, "original_lat"])
		print("Finished getting lat/lon data")
		# print the range of test_lons and test_lats
		print("test_lons.min(): ", test_lons.min())
		print("test_lons.max(): ", test_lons.max())
		print("test_lats.min(): ", test_lats.min())
		print("test_lats.max(): ", test_lats.max())

		# get the upper and lower depth of the test profiles
		test_upper_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		test_lower_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		# check shape
		print("test_upper_depth.shape: ", test_upper_depth.shape)
		print("obs_upper_depth_matrix.shape: ", upper_depth_all.shape)
		test_upper_depth[test_profile_id_num, :] = np.array(upper_depth_all[test_profile_id_num])
		test_lower_depth[test_profile_id_num, :] = np.array(lower_depth_all[test_profile_id_num])
		print("Finished getting depth data")

		# save location data for test profiles
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_profile_id_' + job_id + '.csv', test_profile_id_all, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_lons_' + job_id + '.csv', test_lons, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_lats_' + job_id + '.csv', test_lats, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_upper_depth_' + job_id + '.csv', test_upper_depth, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_lower_depth_' + job_id + '.csv', test_lower_depth, delimiter = ',')

		# initialize the scaled difference
		scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(binn_obs_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or torch.isnan(best_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				simu_soc = best_simu_soc[i, :]
				lower_depth = test_lower_depth[i]
				upper_depth = test_upper_depth[i]
				temp_simu_sum = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or torch.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							# print('outlier: ', test_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu_sum += simu_soc[j] * (upper_depth[j] - lower_depth[j])
						temp_obs_sum += obs_soc[j] * (upper_depth[j] - lower_depth[j])
				scaled_diff[i] = temp_obs_sum/temp_simu_sum
				# # print outlier
				if scaled_diff[i] > 2:
					print('outlier: ', test_profile_id_all[i], scaled_diff[i])

		# Plot the scaled difference
		visualization_utils.plot_observations_world_map(test_lons, test_lats, scaled_diff, PLOT_DIR, "test_scaled_diff_" + job_id, us_only=True)


		############
		# Val Maps #
		############

		
		# get the latitudes and longitudes of the validation profiles by matching ProfileID in env_info with the val_profile_id
		val_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
		val_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
		val_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
		val_profile_id_all[val_profile_id] = val_profile_id
		val_profile_id_num = val_profile_id.numpy().astype(int)
		val_lons[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lon"])
		val_lats[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lat"])
		# print the range of val_lons and val_lats
		print("val_lons.min(): ", val_lons.min())
		print("val_lons.max(): ", val_lons.max())
		print("val_lats.min(): ", val_lats.min())
		print("val_lats.max(): ", val_lats.max())

		# get the upper and lower depth of the validation profiles
		val_upper_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		val_lower_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		val_upper_depth[val_profile_id_num, :] = np.array(upper_depth_all[val_profile_id_num])
		val_lower_depth[val_profile_id_num, :] = np.array(lower_depth_all[val_profile_id_num])

		# save location data for validation profiles
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_profile_id_' + job_id + '.csv', val_profile_id_all, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_lons_' + job_id + '.csv', val_lons, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_lats_' + job_id + '.csv', val_lats, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_upper_depth_' + job_id + '.csv', val_upper_depth, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_lower_depth_' + job_id + '.csv', val_lower_depth, delimiter = ',')

		# Plot maps to show the scaled difference between the predicted and observed SOC values for validation profiles
		# convert nan to 0
		# binn_obs_soc[np.isnan(binn_obs_soc)] = 0
		# best_simu_soc[torch.isnan(best_simu_soc)] = 0
		# initialize the scaled difference
		scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(val_simu_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or torch.isnan(val_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				# print(obs_soc)
				# print(obs_soc.dtype)
				simu_soc = val_simu_soc[i, :]
				lower_depth = val_lower_depth[i]
				upper_depth = val_upper_depth[i]
				temp_simu = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or torch.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							# print('outlier: ', val_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu += simu_soc[j] * (upper_depth[j] - lower_depth[j])
						temp_obs_sum += obs_soc[j] * (upper_depth[j] - lower_depth[j])
				scaled_diff[i] = temp_obs_sum/temp_simu
				# print outlier
				if scaled_diff[i] > 2:
					print('outlier: ', val_profile_id_all[i], scaled_diff[i])


		# Plot the scaled difference
		visualization_utils.plot_observations_world_map(val_lons, val_lats, scaled_diff, PLOT_DIR, "validation_scaled_diff_" + job_id, us_only=True)


		##############
		# Train Maps #
		##############

		# get the latitudes and longitudes of the training profiles by matching ProfileID in env_info with the train_profile_id
		train_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
		train_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
		train_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
		train_profile_id_all[train_profile_id] = train_profile_id
		train_profile_id_num = train_profile_id.numpy().astype(int)
		train_lons[train_profile_id_num] = np.array(env_info.loc[train_profile_id_num, "original_lon"])
		train_lats[train_profile_id_num] = np.array(env_info.loc[train_profile_id_num, "original_lat"])
		# print the range of lon and lat in the env_info
		print("env_info.lon.min(): ", env_info.loc[train_profile_id_num, "original_lon"].max())
		print("env_info.lon.max(): ", env_info.loc[train_profile_id_num, "original_lon"].min())
		print("env_info.lat.min(): ", env_info.loc[train_profile_id_num, "original_lat"].max())
		print("env_info.lat.max(): ", env_info.loc[train_profile_id_num, "original_lat"].min())
		

		# get the upper and lower depth of the training profiles
		train_upper_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		train_lower_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		train_upper_depth[train_profile_id_num, :] = np.array(upper_depth_all[train_profile_id_num])
		train_lower_depth[train_profile_id_num, :] = np.array(lower_depth_all[train_profile_id_num])

		# save location data for training profiles
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_profile_id_' + job_id + '.csv', train_profile_id_all, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lons_' + job_id + '.csv', train_lons, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lats_' + job_id + '.csv', train_lats, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_upper_depth_' + job_id + '.csv', train_upper_depth, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lower_depth_' + job_id + '.csv', train_lower_depth, delimiter = ',')


		# Plot maps to show the scaled difference between the predicted and observed SOC values for training profiles
		scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(train_simu_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or torch.isnan(train_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				simu_soc = train_simu_soc[i, :]
				lower_depth = train_lower_depth[i]
				upper_depth = train_upper_depth[i]
				temp_simu = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or torch.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							# print('outlier: ', train_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu += simu_soc[j] * (upper_depth[j] - lower_depth[j])
						temp_obs_sum += obs_soc[j] * (upper_depth[j] - lower_depth[j])
				scaled_diff[i] = temp_obs_sum/temp_simu
				# print outlier
				if scaled_diff[i] > 2:
					print('outlier: ', train_profile_id_all[i], scaled_diff[i])

		# Plot the scaled difference
		visualization_utils.plot_observations_world_map(train_lons, train_lats, scaled_diff, PLOT_DIR, "train_scaled_diff_" + job_id, us_only=True)


		print("-----------------Model Test Finished at " + str(datetime.now()) + "-----------------")

		# Predict the SOC values based on Grid environmental information using the best model
		grid_simu_soc, grid_pred_para = best_guess_model(torch.tensor(predict_data_x, device=device), torch.tensor(predict_data_z, device=device),
														torch.tensor(predict_data_c, device=device), whether_predict = 1)
		# Save the predicted SOC values, parameters and location data into csv files
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_simu_soc_' + job_id + '.csv', grid_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_pred_para_' + job_id + '.csv', grid_pred_para.detach().cpu().numpy(), delimiter = ',')
		# save grid_env_info_US['Original_Lat'] and grid_env_info_US['Original_Lon'] to csv files
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_lons_' + job_id + '.csv', grid_env_info_US['original_lon'], delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_lats_' + job_id + '.csv', grid_env_info_US['original_lat'], delimiter = ',')


		# Bulk simulation for the grid data
		if args.model != 'nn_only':
			carbon_input_pred, cpool_steady_state_pred, cpools_layer_pred, soc_layer_pred, total_res_time_pred, \
				total_res_time_base_pred, res_time_base_pools_pred, t_scaler_pred, bulk_A_pred, \
				w_scaler_pred, bulk_K_pred, bulk_V_pred, bulk_xi_pred, bulk_I_pred, litter_fraction_pred = fun_bulk_simu(grid_pred_para.to(device), \
																												torch.tensor(predict_data_x, device=device), args.vertical_mixing, args.vectorized)

			# Save the bulk simulation results into csv files
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_carbon_input_' + job_id + '.csv', carbon_input_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_soc_layer_' + job_id + '.csv', soc_layer_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_total_res_time_' + job_id + '.csv', total_res_time_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_t_scaler_' + job_id + '.csv', t_scaler_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_A_' + job_id + '.csv', bulk_A_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_w_scaler_' + job_id + '.csv', w_scaler_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_K_' + job_id + '.csv', bulk_K_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_V_' + job_id + '.csv', bulk_V_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_I_' + job_id + '.csv', bulk_I_pred.detach().cpu().numpy(), delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_pred.detach().cpu().numpy(), delimiter = ',')


		print("-----------------Model Prediction Finished at " + str(datetime.now()) + "-----------------")

		# FINAL SUMMARY MAPS
		# Scatters of true-vs-predicted SOC (grid).
		# Each row represents a layer (or all layers), each column represents a split (train/val/test)
		titles = ["Train: All Depths", "Val: All Depths", "Test: All Depths"]
		y_hats = [best_guess_train_y_hat.flatten(), best_guess_val_y_hat.flatten(), best_guess_test_y_hat.flatten()]  # predictions
		ys = [train_y.flatten(), val_y.flatten(), test_y.flatten()]  # labels
		LAYER_BOUNDARIES = [0, 0.1, 0.3, 1.0, 50.0]
		for i in range(len(LAYER_BOUNDARIES) - 1):  # Loop through layers
			layer_loc_train = (train_z >= LAYER_BOUNDARIES[i]) & (train_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			layer_loc_val = (val_z >= LAYER_BOUNDARIES[i]) & (val_z < LAYER_BOUNDARIES[i+1]) 
			layer_loc_test = (test_z >= LAYER_BOUNDARIES[i]) & (test_z < LAYER_BOUNDARIES[i+1]) 
			y_hats.extend([best_guess_train_y_hat[layer_loc_train], best_guess_val_y_hat[layer_loc_val], best_guess_test_y_hat[layer_loc_test]])
			ys.extend([train_y[layer_loc_train], val_y[layer_loc_val], test_y[layer_loc_test]])
			layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
			titles.extend([f'Train: {layer_str}', f'Val: {layer_str}', f'Test: {layer_str}'])
		visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"FINAL_scatters.png"), y_hats, ys, titles, cols=3)

		# Maps of true-vs-predicted SOC (grid)
		# Each row represents a layer, each column represents a split (train/val/test) and {true or predicted}
		lons_list = []
		lats_list = []
		values_list = []
		vars_list = []
		for i in range(len(LAYER_BOUNDARIES) - 1):
			# For each site: compute average SOC over observations in this layer
			layer_loc_train = (train_z >= LAYER_BOUNDARIES[i]) & (train_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			train_true_soc = torch.where(layer_loc_train, train_y, torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
			train_true_soc = torch.nanmean(train_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
			train_pred_soc = torch.where(layer_loc_train.to(device), best_guess_train_y_hat, torch.nan)  # Same for predictions
			train_pred_soc = torch.nanmean(train_pred_soc, dim=1)

			# Repeat above for val data
			layer_loc_val = (val_z >= LAYER_BOUNDARIES[i]) & (val_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			val_true_soc = torch.where(layer_loc_val, val_y, torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
			val_true_soc = torch.nanmean(val_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
			val_pred_soc = torch.where(layer_loc_val.to(device), best_guess_val_y_hat, torch.nan)  # Same for predictions
			val_pred_soc = torch.nanmean(val_pred_soc, dim=1)

			# Repeat above for test data
			layer_loc_test = (test_z >= LAYER_BOUNDARIES[i]) & (test_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			test_true_soc = torch.where(layer_loc_test, test_y, torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
			test_true_soc = torch.nanmean(test_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
			test_pred_soc = torch.where(layer_loc_test.to(device), best_guess_test_y_hat, torch.nan)  # Same for predictions
			test_pred_soc = torch.nanmean(test_pred_soc, dim=1)

			# Collect results
			lons_list.extend([train_c[:, 0], train_c[:, 0], val_c[:, 0], val_c[:, 0], test_c[:, 0], test_c[:, 0]])
			lats_list.extend([train_c[:, 1], train_c[:, 1], val_c[:, 1], val_c[:, 1], test_c[:, 1], test_c[:, 1]])
			values_list.extend([train_true_soc, train_pred_soc, val_true_soc, val_pred_soc, test_true_soc, test_pred_soc])
			layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
			vars_list.extend([f'True SOC - Train: {layer_str}', f'Predicted SOC - Train: {layer_str}',
								f'True SOC - Val: {layer_str}', f'Predicted SOC - Val: {layer_str}',
								f'True SOC - Test: {layer_str}', f'Predicted SOC - Test: {layer_str}',])
		visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"FINAL_soc_maps.png"),
				lons_list, lats_list, values_list, vars_list, us_only=True, cols=6)

		# Parameter maps. Each row is a parameter, each column represents a split (train/val/test/grid)
		if args.model != "nn_only":
			# Use PRODA parameters as "labels" to compare with our predicted parameters
			lons_list = []
			lats_list = []
			values_list = []
			vars_list = []
			for para_idx in range(model_without_ddp.num_params):
				lons_list.extend([train_c[:, 0], train_c[:, 0], val_c[:, 0], val_c[:, 0], test_c[:, 0], test_c[:, 0], predict_data_c[:, 0], predict_data_c[:, 0]])
				lats_list.extend([train_c[:, 1], train_c[:, 1], val_c[:, 1], val_c[:, 1], test_c[:, 1], test_c[:, 1], predict_data_c[:, 1], predict_data_c[:, 1]])
				values_list.extend([train_proda_para[:, para_idx], best_guess_train_pred_para[:, para_idx],
									val_proda_para[:, para_idx], best_guess_val_pred_para[:, para_idx],
									test_proda_para[:, para_idx], best_guess_test_pred_para[:, para_idx],
									grid_PRODA_para[:, para_idx], grid_pred_para[:, para_idx]])
				para_name = para_names[para_idx]
				vars_list.extend([f'PRODA para {para_name} - Train', f'Predicted para {para_name} - Train',
									f'PRODA para {para_name} - Val', f'Predicted para {para_name} - Val',
									f'PRODA para {para_name} - Test', f'Predicted para {para_name} - Test',
									f'PRODA para {para_name} - Grid', f'Predicted para {para_name} - Grid'])
			visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"FINAL_para_maps.png"),
					lons_list, lats_list, values_list, vars_list, us_only=True, cols=4)

			# Also plot scatters
			y_hats = []
			ys = []
			titles = []
			for para_idx in range(model_without_ddp.num_params):
				y_hats.extend([best_guess_train_pred_para[:, para_idx], best_guess_val_pred_para[:, para_idx], best_guess_test_pred_para[:, para_idx], grid_pred_para[:, para_idx]])
				ys.extend([train_proda_para[:, para_idx], val_proda_para[:, para_idx], test_proda_para[:, para_idx], grid_PRODA_para[:, para_idx]])
				para_name = para_names[para_idx]
				titles.extend([f'Train: {para_name}', f'Val: {para_name}', f'Test: {para_name}', f'Grid: {para_name}'])
			visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"FINAL_para_scatters.png"), y_hats, ys, titles, cols=2)


	# end if rank == 0:
	else:
		if whether_break.item() == 1:
			print("Rank {} finished".format(rank))
			return
		
		
	# Pause to allow rank 0 to finish writing the summary file
	dist.barrier()
	print("Rank {} finished".format(rank))
	dist.destroy_process_group()



if __name__ == '__main__':

	# Number of CPUs requester
	world_size = args.num_CPU
	processes = []

	# Create job ID
	job_id = create_output_folders(args)
	print("MAIN, JOB ID", job_id)

	# Spawn method is required if using GPU
	if torch.cuda.is_available():
		assert world_size == len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), "If using GPU: world_size (num_CPU) must equal number of GPUs in CUDA_VISIBLE_DEVICES"
		import torch.multiprocessing as mp
		mp.set_start_method('spawn', force=True)

	# Create the processes
	for rank in range(world_size):
		p = Process(target=worker, args=(rank, world_size, job_id))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	
