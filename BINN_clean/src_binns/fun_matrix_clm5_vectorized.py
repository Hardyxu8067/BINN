import time
import numpy as np
import torch
import traceback
import math


def fun_model_simu(tensor_para, tensor_frocing_steady_state, tensor_obs_layer_depth, vertical_mixing, vectorized='true'):
	"""
	Simulate the soil carbon profile using the CLM5 model at the depth of the observations
	"""
	device = tensor_para.device
	# convert tensor to numpy
	para = tensor_para
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state 
	obs_layer_depth = tensor_obs_layer_depth

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)
	

	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = para.shape[0]
	simu_ouput = (torch.ones((profile_num, 200))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_para = para[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]
		profile_obs_layer_depth = obs_layer_depth[iprofile, :]
		valid_layer_loc = torch.where(torch.isnan(profile_obs_layer_depth) == False)[0]

		if torch.isnan(torch.sum(profile_para)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:
			
			# print(profile_para)
			# model simulation
			profile_simu_soc = fun_matrix_clm5(profile_para, profile_force_steady_state, vertical_mixing, vectorized)
			
			for ilayer in range(0, len(valid_layer_loc)):
				layer_depth = profile_obs_layer_depth[valid_layer_loc[ilayer]]
				depth_diff = zsoi[0:n_soil_layer] - layer_depth
				if len(torch.where(depth_diff == 0)[0]) == 0:
					if depth_diff[0] > 0:
						node_depth_upper_loc = 0
						node_depth_lower_loc = 0
					elif depth_diff[-1] < 0:
						node_depth_upper_loc = n_soil_layer - 1
						node_depth_lower_loc = n_soil_layer - 1
					else:
						node_depth_upper_loc = torch.where(depth_diff[:-1]*depth_diff[1:]<0)[0]
						node_depth_lower_loc = node_depth_upper_loc + 1
					# end if depth_diff[0] > 0:
				else:
					node_depth_upper_loc = torch.where(depth_diff == 0)
					node_depth_lower_loc = node_depth_upper_loc
				#end if len(torch.where(depth_diff == 0)[0]) == 0:
				if node_depth_lower_loc == node_depth_upper_loc:
					simu_ouput[iprofile, valid_layer_loc[ilayer]] = profile_simu_soc[node_depth_lower_loc]
				else:
					simu_ouput[iprofile, valid_layer_loc[ilayer]] = \
					profile_simu_soc[node_depth_lower_loc] \
					+ (profile_simu_soc[node_depth_upper_loc] - profile_simu_soc[node_depth_lower_loc]) \
					/(zsoi[node_depth_upper_loc] - zsoi[node_depth_lower_loc]) \
					*(layer_depth - zsoi[node_depth_lower_loc])
			# end for
		# end if 
	#end for iprofile
	# print("fun_model_simu", time.time()-start_time)
	return simu_ouput
	
# end def fun_model_simu

def fun_model_prediction(tensor_para, tensor_frocing_steady_state, vertical_mixing, vectorized='true'):
	"""
	Predict soil carbon profiles using the CLM5 model, at the 20 fixed CLM5 layers
	"""
	device = tensor_para.device
	# convert tensor to numpy
	para = tensor_para
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state 

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)
	

	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = para.shape[0]
	simu_ouput = (torch.ones((profile_num, 200))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_para = para[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]

		if torch.isnan(torch.sum(profile_para)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:
			
			# print(profile_para)
			# model simulation
			profile_simu_soc = fun_matrix_clm5(profile_para, profile_force_steady_state, vertical_mixing, vectorized)
			
			# save simulation results
			simu_ouput[iprofile, 0:20] = profile_simu_soc

	#end for iprofile
	return simu_ouput

# Sensitivity analysis of the soil carbon profile using the CLM5 model
def fun_model_sensitivity(tensor_para, tensor_frocing_steady_state, vertical_mixing, vectorized='true'):
	device = tensor_para.device
	# convert tensor to numpy
	para = tensor_para
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state 

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)
	

	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = frocing_steady_state.shape[0]
	simu_ouput = (torch.ones((profile_num, 20))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_para = para[:]
		# print(profile_para)
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]
		# check if the input is valid
		# print(torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])))
		# print(torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])))


		if torch.isnan(torch.sum(profile_para)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:
			
			# print(profile_para)
			# model simulation
			profile_simu_soc = fun_matrix_clm5(profile_para, profile_force_steady_state, vertical_mixing, vectorized)
			

			# save simulation results
			simu_ouput[iprofile, 0:20] = profile_simu_soc

	#end for iprofile
	return simu_ouput
	
# end def fun_model_simu





#######################################################
# forward simulation for clm5
#######################################################
def fun_matrix_clm5(para, frocing_steady_state, vertical_mixing, vectorized='true'):
	device = para.device
	#---------------------------------------------------
	# offical starting simulation
	#---------------------------------------------------
	use_beta = 1
	normalize_q10_to_century_tfunc = False
	global timestep_num, season_num, month_num, \
		kelvin_to_celsius, use_vertsoilc, npool, npool_vr, n_soil_layer, \
		days_per_year, secspday, days_per_month, days_per_season, days_per_timestep
		
	season_num = 4
	month_num = 12
	kelvin_to_celsius = 273.15
	use_vertsoilc = 1
	npool = 7
	npool_vr = 140
	n_soil_layer = 20
	days_per_year = 365
	secspday = 24*60*60
	days_per_month = torch.tensor([[31, 30, 31, 28, 31, 30, 31, 31, 30, 31, 30, 31]]).transpose(0, 1).to(device)
	days_per_season = torch.tensor([[92, 89, 92, 92]]).transpose(0, 1).to(device)
	
	timestep_num = month_num
	days_per_timestep = days_per_season

	global max_altdepth_cryoturbation, max_depth_cryoturb
	max_altdepth_cryoturbation = 2
	max_depth_cryoturb = 3

	global dz, zisoi, zisoi_0, zsoi, dz_node
	# width between two interfaces
	dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
		8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
		0.200000000000000, 0.240000000000000, 0.280000000000000, \
		0.320000000000000, 0.360000000000000, 0.400000000000000, \
		0.440000000000000, 0.540000000000000, 0.640000000000000, \
		0.740000000000000, 0.840000000000000, 0.940000000000000, \
		1.04000000000000, 1.14000000000000, 2.39000000000000, \
		4.67553390593274, 7.63519052838329, 11.1400000000000, \
		15.1154248593737]).to(device)
	
	# depth of the interface
	zisoi = torch.tensor([2.000000000000000E-002, 6.000000000000000E-002, \
		0.120000000000000, 0.200000000000000, 0.320000000000000, \
		0.480000000000000, 0.680000000000000, 0.920000000000000, \
		1.20000000000000, 1.52000000000000, 1.88000000000000, \
		2.28000000000000, 2.72000000000000, 3.26000000000000, \
		3.90000000000000, 4.64000000000000, 5.48000000000000, \
		6.42000000000000, 7.46000000000000, 8.60000000000000, \
		10.9900000000000, 15.6655339059327, 23.3007244343160, \
		34.4407244343160, 49.5561492936897]).to(device)

	zisoi_0 = 0;

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)

	# depth between two node
	dz_node = zsoi - torch.cat((torch.tensor([0.0]).to(device), zsoi[:-1]), axis = 0)

	# construct a diagonal matrix that contains dz for each layer (20 layers) and 7 pools
	dz_matrix = torch.diag(-1*torch.ones(npool_vr)).to(device)
	# fill the diagonal matrix with dz for each pool (7 pools)
	dz_matrix.diagonal()[0:20] = dz[0:20]
	dz_matrix.diagonal()[20:40] = dz[0:20]
	dz_matrix.diagonal()[40:60] = dz[0:20]
	dz_matrix.diagonal()[60:80] = dz[0:20]
	dz_matrix.diagonal()[80:100] = dz[0:20]
	dz_matrix.diagonal()[100:120] = dz[0:20]
	dz_matrix.diagonal()[120:140] = dz[0:20]
	dz_matrix_diagonal = dz_matrix.diagonal().view(npool_vr, 1)

	
	#---------------------------------------------------
	# steady state forcing
	#---------------------------------------------------
	input_vector_cwd_steady_state = frocing_steady_state[0:12, 0, 1]
	input_vector_litter1_steady_state = frocing_steady_state[0:12, 0, 2]
	input_vector_litter2_steady_state = frocing_steady_state[0:12, 0, 3]
	input_vector_litter3_steady_state = frocing_steady_state[0:12, 0, 4]
	altmax_lastyear_profile_steady_state = frocing_steady_state[0:12, 0, 5]
	altmax_current_profile_steady_state = frocing_steady_state[0:12, 0, 6]
	nbedrock_steady_state = frocing_steady_state[0:12, 0, 7].type(torch.int)
	
	xio_steady_state = frocing_steady_state[0:20, 0:12, 8]
	xin_steady_state = frocing_steady_state[0:20, 0:12, 9]
	sand_vector_steady_state = frocing_steady_state[0:20, 0:12, 10]
	soil_temp_profile_steady_state = frocing_steady_state[0:20, 0:12, 11]
	soil_water_profile_steady_state = frocing_steady_state[0:20, 0:12, 12]
	
	#---------------------------------------------------
	# define parameters to be optimised
	#---------------------------------------------------
	if vertical_mixing == 'original':
		# diffusion (bioturbation) 10^(-4) (m2/yr)
		bio = para[0]*(5*1e-4 - 3*1e-5) + 3*1e-5
		# cryoturbation 5*10^(-4) (m2/yr)
		cryo = para[1]*(16*1e-4 - 3*1e-5) + 3*1e-5
	else:
		# use the alternative tri-matrix
		slope = para[0]*((0) - (-3)) + (-3)  # increase, decrease, or no change of diffusion rate with depth
		# intercept = para[1]*((-6) - (-10)) + (-10)
		intercept = para[1]*((-2) - (-12)) + (-12)

	#  Q10 (unitless) 1.5
	q10 = para[2]*(3 - 1.2) + 1.2
	# Q10 when forzen (unitless) 1.5
	fq10 = q10
	# parameters used in vertical discretization of carbon inputs 10 (metre)
	# efolding = para[3]*(1 - 0.0001) + 0.0001
	efolding = para[3]*(1 - 0.1) + 0.1
	# turnover time of CWD (yr) 3.3333
	tau4cwd = para[4]*(6 - 1) + 1
	# tau for metabolic litter (yr) 0.0541
	tau4l1 = para[5]*(0.11 - 0.0001) + 0.0001
	# tau for cellulose litter (yr) 0.2041
	tau4l2 = para[6]*(0.3 - 0.1) + 0.1
	# tau for lignin litter (yr)
	tau4l3 = tau4l2
	# tau for fast SOC (yr) 0.1370
	tau4s1 = para[7]*(0.5 - 0.0001) + 0.0001
	# tau for slow SOC (yr) 5
	tau4s2 = para[8]*(10 - 1) + 1
	# tau for passive SOC (yr) 222.222
	tau4s3 = para[9]*(400 - 20) + 20
	# fraction from l1 to s2, 0.45
	fl1s1 = para[10]*(0.8 - 0.1) + 0.1
	# fraction from l2 to s1, 0.5
	fl2s1 = para[11]*(0.8 - 0.2) + 0.2
	# fraction from l3 to s2, 0.5
	fl3s2 = para[12]*(0.8 - 0.2) + 0.2
	# fraction from s1 to s2, sand dependeted
	fs1s2 = para[13]*(0.4 - 0.0001) + 0.0001
	# fraction from s1 to s3, sand dependeted
	fs1s3 = para[14]*(0.1 - 0.0001) + 0.0001
	# fraction from s2 to s1, 0.42
	fs2s1 = para[15]*(0.74 - 0.1) + 0.1
	# fraction from s2 to s3, 0.03
	fs2s3 = para[16]*(0.1 - 0.0001) + 0.0001
	# fraction from s3 to s1, 0.45
	fs3s1 = para[17]*(0.9 - 0.0001) + 0.0001
	# fraction from cwd to l2, 0.76
	fcwdl2 = para[18]*(1 - 0.5) + 0.5
	
	# water scaling factor
	w_scaling = para[19]*(5 - 0.0001) + 0.0001
	# beta to describe the shape of vertical profile
	# beta = 0.95
	# or fix it at first ~ 0.6/0.7
	# beta = para[20]*(0.9 - 0.5) + 0.5
	beta = para[20] *(0.9999 - 0.5) + 0.5
	# beta = 0.7 *(0.9 - 0.5) + 0.5
	# beta = 0.8

	# Separate intercept for "leach" (downward transport)
	if vertical_mixing == 'simple_two_intercepts':
		intercept_leach = para[21]*((-2) - (-12)) + (-12)
	elif vertical_mixing == 'simple_one_intercept':
		intercept_leach = intercept

	# maximum and minimum water potential (MPa)
	maxpsi= -0.0020
	minpsi= -2; # minimum water potential (MPa)
	adv = 0 # parameter for advection (m/yr)

	#####################################################
	# steady state solutions
	#####################################################

	#---------------------------------------------------
	# steady state env scalar
	#---------------------------------------------------
	xiw = (torch.ones(n_soil_layer, timestep_num)*np.nan).to(device)
	xio = xio_steady_state
	xin = xin_steady_state

	if vectorized in ['false', 'compare']:  # Old way of calculating xit
		xit_old = (torch.ones(n_soil_layer, timestep_num)*np.nan).to(device)
		for itimestep in range(timestep_num):
			# temperature related function xit
			# calculate rate constant scalar for soil temperature
			# assuming that the base rate constants are assigned for non-moisture
			# limiting conditions at 25 C.
			for ilayer in range(n_soil_layer):
				if soil_temp_profile_steady_state[ilayer, itimestep] >= (0 + kelvin_to_celsius):
					xit_old[ilayer, itimestep] = q10**((soil_temp_profile_steady_state[ilayer, itimestep] - (kelvin_to_celsius + 25))/10)
				else:
					xit_old[ilayer, itimestep] = q10**((273.15 - 298.15)/10)*(fq10**((soil_temp_profile_steady_state[ilayer, itimestep] - (0 + kelvin_to_celsius))/10))
				# end if soil_temp_profile[ilayer, itimestep] >= 0 + kelvin_to_celsius:
			# end for layer
			
			catanf_30 = catanf(torch.tensor(30.0).to(device))
			normalization_tref = torch.tensor(15).to(device)
			if normalize_q10_to_century_tfunc == True:
				# scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
				normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10**((normalization_tref-25)/10))
				xit_old[:, itimestep] = xit_old[:, itimestep]*normalization_factor
		xit = xit_old

	if vectorized in ['true', 'compare']:  # New way to calculate xit (vectorized))
		xit_above_freezing = torch.pow(q10, ((soil_temp_profile_steady_state - (kelvin_to_celsius + 25))/10))  # Above freezing case first
		xit_below_freezing = torch.pow(q10, ((273.15 - 298.15)/10)) * torch.pow(fq10, ((soil_temp_profile_steady_state - (0 + kelvin_to_celsius))/10))	
		freezing_mask = (soil_temp_profile_steady_state < (0 + kelvin_to_celsius)).detach().int()  # Create a mask which is True when the soil temperatue is below freezing
		xit = xit_above_freezing * (1-freezing_mask) + xit_below_freezing * freezing_mask  # [freezing_mask] = xit_below_freezing[freezing_mask].clone()
		catanf_30 = catanf(torch.tensor(30.0).to(device))
		normalization_tref = torch.tensor(15).to(device)
		if normalize_q10_to_century_tfunc == True:
			# scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
			normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10**((normalization_tref-25)/10))
			xit = xit * normalization_factor
	
	if vectorized == 'compare':
		assert(torch.allclose(xit_old, xit))

	xiw = soil_water_profile_steady_state*w_scaling
	xiw[xiw > 1] = 1

	#---------------------------------------------------
	# steady state tridiagnal matrix, A matrix, K matrix, fire matrix
	#---------------------------------------------------
	sand_vector = torch.mean(sand_vector_steady_state, axis = 1)

	# allocation matrix
	if vectorized in ['false', 'compare']:
		# a_ma_old_start = time.time()
		a_ma = a_ma_old = a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector)
		# a_ma_old_time = time.time() - a_ma_old_start
	if vectorized in ['true', 'compare']:
		# a_ma_new_start = time.time()
		a_ma = a_matrix_vectorized(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector)
		# a_ma_new_time = time.time() - a_ma_new_start
	if vectorized == 'compare':
		assert torch.allclose(a_ma_old, a_ma)
		# print("A_MA: OLD", a_ma_old_time, "NEW", a_ma_new_time)

	kk_ma_middle = (torch.zeros([npool_vr, npool_vr, timestep_num])*np.nan).to(device) 
	tri_ma_middle = (torch.zeros([npool_vr, npool_vr, timestep_num])*np.nan).to(device) 
	
	for itimestep in range(timestep_num):
		# decomposition matrix
		timesteply_xit = xit[:, itimestep]
		timesteply_xiw = xiw[:, itimestep]
		timesteply_xio = xio[:, itimestep]
		timesteply_xin = xin[:, itimestep]

		# K matrix
		if vectorized in ['false', 'compare']:
			kk_ma = kk_ma_old = kk_matrix(timesteply_xit, timesteply_xiw, timesteply_xio, timesteply_xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3)
		if vectorized in ['true', 'compare']:
			kk_ma = kk_matrix_vectorized(timesteply_xit, timesteply_xiw, timesteply_xio, timesteply_xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3)
		if vectorized in 'compare':
			assert torch.allclose(kk_ma_old, kk_ma)
		kk_ma_middle[:, :, itimestep] = kk_ma

		# tri matrix	
		timesteply_nbedrock = nbedrock_steady_state[itimestep]
		timesteply_altmax_current_profile = altmax_current_profile_steady_state[itimestep]
		timesteply_altmax_lastyear_profile = altmax_lastyear_profile_steady_state[itimestep]

		if vertical_mixing == 'original':
			if vectorized in ['false', 'compare']:
				tri_ma_old_start = time.time()
				tri_ma = tri_ma_old = tri_matrix_old(timesteply_nbedrock, timesteply_altmax_current_profile, timesteply_altmax_lastyear_profile, bio, adv, cryo)
				tri_ma_old_time = time.time() - tri_ma_old_start
			if vectorized in ['true', 'compare']:
				tri_ma_new_start = time.time()
				tri_ma = tri_matrix_old_improved(timesteply_nbedrock, timesteply_altmax_current_profile, timesteply_altmax_lastyear_profile, bio, adv, cryo)
				tri_ma_new_time = time.time() - tri_ma_new_start
		else:
			if vectorized in ['false', 'compare']:
				assert vertical_mixing == 'simple_one_intercept'  # For non-vectorized tri_matrix_alternative, only one intercept is supported
				tri_ma = tri_ma_old = tri_matrix_alternative(timesteply_nbedrock, slope, intercept, device)
			if vectorized in ['true', 'compare']:
				tri_ma = tri_matrix_alternative_vectorized(timesteply_nbedrock, slope, intercept, intercept_leach, device)
		# if vectorized == 'compare':
		# 	# TODO: the tri_ma implementations may give slightly different results?
		# 	print("Tri ma old", tri_ma_old)
		# 	print("Tri ma new", tri_ma)
		# 	assert torch.allclose(tri_ma_old, tri_ma)
		# 	print("TRI MATRIX. OLD", tri_ma_old_time, "NEW", tri_ma_new_time)
		tri_ma_middle[:, :, itimestep] = tri_ma

	# end for itimestep
	tri_ma = torch.mean(tri_ma_middle, axis = 2)
	kk_ma = torch.mean(kk_ma_middle, axis = 2)
	
	#---------------------------------------------------
	# steady state vertical profile, input allocation
	#---------------------------------------------------
	# in the original beta model in Jackson et al 1996, the unit for the depth of the soil is cm (dmax*100)
	m_to_cm = 100

	vertical_prof = (torch.ones(n_soil_layer)*np.nan).to(device) 
	if torch.mean(altmax_lastyear_profile_steady_state) > 0:
		if vectorized in ['false', 'compare']:
			# Old way of calculating vertical_prof
			vertical_prof_old = (torch.ones(n_soil_layer)*np.nan).to(device) 
			for j in range(n_soil_layer): #1:n_soil_layer
				if j == 0: # first layer
					vertical_prof_old[j] = (beta**((zisoi_0)*m_to_cm) - beta**(zisoi[j]*m_to_cm))/dz[j]
				else:
					vertical_prof_old[j] = (beta**((zisoi[j - 1])*m_to_cm) - beta**(zisoi[j]*m_to_cm))/dz[j]
				# end j == 0:
			# end for j in range(n_soil_layer):
			vertical_prof = vertical_prof_old
		if vectorized in ['true', 'compare']:
			# New way to calculate vertical_prof (vectorized)
			vertical_prof = (torch.ones(n_soil_layer)*np.nan).to(device) 
			vertical_prof[0] = (beta**((zisoi_0)*m_to_cm) - beta**(zisoi[0]*m_to_cm))/dz[0]
			vertical_prof[1:n_soil_layer] = (beta**((zisoi[0:n_soil_layer-1])*m_to_cm) - beta**(zisoi[1:n_soil_layer]*m_to_cm))/dz[1:n_soil_layer]
		if vectorized == 'compare':
			assert torch.allclose(vertical_prof_old, vertical_prof)

	else:
		vertical_prof[0] = 1/dz[0]
		vertical_prof[1:] = 0
	# end if np.mean(altmax_lastyear_profile_steady_state) > 0:

	vertical_input = dz[0:n_soil_layer]*vertical_prof/sum(vertical_prof*dz[0:n_soil_layer])
	
	#---------------------------------------------------
	# steady state analytical solution of soc
	#---------------------------------------------------
	matrix_in = (torch.ones([npool_vr, 1])*np.nan).to(device) 
	# total input amount
	input_tot_cwd = torch.nansum(input_vector_cwd_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter1 = torch.nansum(input_vector_litter1_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter2 = torch.nansum(input_vector_litter2_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter3 = torch.nansum(input_vector_litter3_steady_state)/days_per_year # (gc/m2/day)

	# redistribution by beta
	matrix_in[0:20, 0] = input_tot_cwd*vertical_input/dz[0:n_soil_layer] # litter input gc/m3/day
	matrix_in[20:40, 0] = input_tot_litter1*vertical_input/dz[0:n_soil_layer]
	matrix_in[40:60, 0] = input_tot_litter2*vertical_input/dz[0:n_soil_layer]
	matrix_in[60:80, 0] = input_tot_litter3*vertical_input/dz[0:n_soil_layer]
	matrix_in[80:140, 0] = 0
	
	# analytical solution of soc pools
	try:
		cpool_steady_state = torch.linalg.solve((torch.matmul(a_ma, kk_ma)- tri_ma), (-matrix_in))
	except Exception:
		traceback.print_exc()
		print("Predicted Parameters: ", para)
		# check if the matrix is singular and print the matrix
		# check a_ma
		if torch.isnan(torch.sum(a_ma)):
			print("a_ma contains nan")
		if torch.det(a_ma) == 0:
			print("a_ma is singular")
		# check kk_ma
		if torch.isnan(torch.sum(kk_ma)):
			print("kk_ma contains nan")
		if torch.det(kk_ma) == 0:
			print("kk_ma is singular")
			print(torch.diagonal(kk_ma))
		# check tri_ma
		if torch.isnan(torch.sum(tri_ma)):
			print("tri_ma contains nan")
		if torch.det(tri_ma) == 0:
			print("tri_ma is singular")
			print(torch.diagonal(tri_ma, offset=0))
			print(torch.diagonal(tri_ma, offset=1))
			print(torch.diagonal(tri_ma, offset=-1))
		# check matrix_in
		if torch.isnan(torch.sum(matrix_in)):
			print("matrix_in contains nan")
		if torch.det(torch.matmul(a_ma, kk_ma)-tri_ma) == 0:
			print("a_ma*kk_ma - tri_ma is singular")

		cpool_steady_state = (torch.ones([140, 1])*(-1.0)).to(device)*torch.sum(para)/torch.sum(para)
	# end try

	soc_layer = torch.cat((cpool_steady_state[80:100, :], cpool_steady_state[100:120, :], cpool_steady_state[120:140, :]), dim = 1)
	soc_layer = torch.sum(soc_layer, axis = 1) # unit gC/m3
	outcome = soc_layer
	return outcome

#end def fun_forward_simu_clm5


##################################################
# sub-function in matrix equation
##################################################

def a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector):
	device = fl1s1.device
	nlevdecomp = n_soil_layer
	nspools = npool
	nspools_vr = npool_vr
	# a diagnal matrix
	a_ma_vr = torch.diag(-1*torch.ones(nspools_vr)).to(device)

	fcwdl3 = 1 - fcwdl2
	
	transfer_fraction = [fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, fcwdl3]
	for j in range(nlevdecomp):
		a_ma_vr[(3-1)*nlevdecomp+j,(1-1)*nlevdecomp+j] = transfer_fraction[8]
		a_ma_vr[(4-1)*nlevdecomp+j,(1-1)*nlevdecomp+j] = transfer_fraction[9]
		a_ma_vr[(5-1)*nlevdecomp+j,(2-1)*nlevdecomp+j] = transfer_fraction[0]
		a_ma_vr[(5-1)*nlevdecomp+j,(3-1)*nlevdecomp+j] = transfer_fraction[1]
		a_ma_vr[(5-1)*nlevdecomp+j,(6-1)*nlevdecomp+j] = transfer_fraction[5]
		a_ma_vr[(5-1)*nlevdecomp+j,(7-1)*nlevdecomp+j] = transfer_fraction[7]
		a_ma_vr[(6-1)*nlevdecomp+j,(4-1)*nlevdecomp+j] = transfer_fraction[2]
		a_ma_vr[(6-1)*nlevdecomp+j,(5-1)*nlevdecomp+j] = transfer_fraction[3]
		a_ma_vr[(7-1)*nlevdecomp+j,(5-1)*nlevdecomp+j] = transfer_fraction[4]
		a_ma_vr[(7-1)*nlevdecomp+j,(6-1)*nlevdecomp+j] = transfer_fraction[6]
	# end for ilayer
	return a_ma_vr
# end def a_matrix


# If a_ma is a block matrix where each A_{ij} is (nlevdecomp x nlevdecomp),
# fills in the diagonal of block A_{ij} with "value".
# "value" is assumed to be a tensor with a single element.
# For consistency with the paper (Lu et al. 2020), i and j are indexed from 1.
# Modifies a_ma in place.
def fill_submatrix_diagonal(a_ma, nlevdecomp, i, j, value):
	a_ma[range((i-1)*nlevdecomp, i*nlevdecomp), range((j-1)*nlevdecomp, j*nlevdecomp)] = value  # diag_vector


def a_matrix_vectorized(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector):
	device = fl1s1.device
	nlevdecomp = n_soil_layer
	nspools = npool
	nspools_vr = npool_vr

	# a diagonal matrix
	a_ma_vr = torch.diag(-1*torch.ones(nspools_vr, device=device))

	fcwdl3 = 1 - fcwdl2
	transfer_fraction = [fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, fcwdl3]
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 3, 1, transfer_fraction[8])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 4, 1, transfer_fraction[9])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 2, transfer_fraction[0])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 3, transfer_fraction[1])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 6, transfer_fraction[5])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 7, transfer_fraction[7])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 6, 4, transfer_fraction[2])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 6, 5, transfer_fraction[3])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 7, 5, transfer_fraction[4])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 7, 6, transfer_fraction[6])
	return a_ma_vr



def kk_matrix(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3):
	device = xit.device
	
	n_soil_layer = 20
	days_per_year = 365
	# env scalars
	n_scalar = xin[:n_soil_layer] # nitrogen 
	t_scalar = xit[:n_soil_layer] # temperature
	w_scalar = xiw[:n_soil_layer] # water
	o_scalar = xio[:n_soil_layer] # oxygen
	
	# decomposition rate
	kl1 = 1/(days_per_year * tau4l1)
	kl2 = 1/(days_per_year  * tau4l2)
	kl3 = 1/(days_per_year  * tau4l3)
	ks1 = 1/(days_per_year  * tau4s1)
	ks2 = 1/(days_per_year  * tau4s2)
	ks3 = 1/(days_per_year  * tau4s3)
	kcwd = 1/(days_per_year  * tau4cwd)

	decomp_depth_efolding = efolding

	depth_scalar = torch.exp(-zsoi/decomp_depth_efolding)
	xi_tw = t_scalar*w_scalar*o_scalar
	
	kk_ma_vr = (torch.zeros([npool_vr, npool_vr])).to(device)
	for j in range(n_soil_layer):
		kk_ma_vr[0*n_soil_layer+j, 0*n_soil_layer+j] = kcwd * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[1*n_soil_layer+j, 1*n_soil_layer+j] = kl1 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[2*n_soil_layer+j, 2*n_soil_layer+j] = kl2 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[3*n_soil_layer+j, 3*n_soil_layer+j] = kl3 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[4*n_soil_layer+j, 4*n_soil_layer+j] = ks1 * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[5*n_soil_layer+j, 5*n_soil_layer+j] = ks2 * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[6*n_soil_layer+j, 6*n_soil_layer+j] = ks3 * xi_tw[j] * depth_scalar[j]
	# end for j
	return kk_ma_vr
# end def kk_matrix


def kk_matrix_vectorized(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3):
	device = xit.device
	
	n_soil_layer = 20
	days_per_year = 365
	# env scalars
	n_scalar = xin[:n_soil_layer] # nitrogen 
	t_scalar = xit[:n_soil_layer] # temperature
	w_scalar = xiw[:n_soil_layer] # water
	o_scalar = xio[:n_soil_layer] # oxygen
	
	# decomposition rate
	kl1 = 1/(days_per_year * tau4l1)
	kl2 = 1/(days_per_year  * tau4l2)
	kl3 = 1/(days_per_year  * tau4l3)
	ks1 = 1/(days_per_year  * tau4s1)
	ks2 = 1/(days_per_year  * tau4s2)
	ks3 = 1/(days_per_year  * tau4s3)
	kcwd = 1/(days_per_year  * tau4cwd)

	decomp_depth_efolding = efolding

	depth_scalar = torch.exp(-zsoi/decomp_depth_efolding)[:n_soil_layer]  # NOTE added the [:n_soil_layer]
	xi_tw = t_scalar*w_scalar*o_scalar

	diagonal_vector = torch.concatenate([kcwd * xi_tw * depth_scalar,
									     kl1 * xi_tw * depth_scalar * n_scalar,
										 kl2 * xi_tw * depth_scalar * n_scalar,
										 kl3 * xi_tw * depth_scalar * n_scalar,
										 ks1 * xi_tw * depth_scalar,
										 ks2 * xi_tw * depth_scalar,
										 ks3 * xi_tw * depth_scalar], dim=0).to(device)
	assert diagonal_vector.shape == (140, )
	return torch.diag(diagonal_vector)
	# return kk_ma_vr

						
# only diffusion
def tri_matrix_alternative(nbedrock, slope, intercept, device):
	# slope = -1.2
	# intercept = -4
	rate_to_atmos = -0. # # at the surface, part of the CO2 should be released to atmos
	transport_rate = -10**(intercept + slope*torch.log10(zsoi[0:20]))
	transport_rate[nbedrock:] = -10**(-30)
	tri_ma_middle = torch.zeros(n_soil_layer, n_soil_layer, device=device)
	for ilayer in range(n_soil_layer):
		if ilayer == 0:
			tri_ma_middle[ilayer, ilayer] = -1*(transport_rate[ilayer] + rate_to_atmos) # at the surface, part of the CO2 should be released to atmos
			tri_ma_middle[ilayer, (ilayer+1)] = transport_rate[ilayer]
		elif ilayer == (n_soil_layer-1):
			tri_ma_middle[ilayer, ilayer] = -1*transport_rate[ilayer]
			tri_ma_middle[ilayer, (ilayer-1)] = transport_rate[ilayer]
		else:
			tri_ma_middle[ilayer, ilayer] = -2*transport_rate[ilayer]
			tri_ma_middle[ilayer, (ilayer-1)] = transport_rate[ilayer]
			tri_ma_middle[ilayer, (ilayer+1)] = transport_rate[ilayer]
	#end for ilayer in range(n_soil_layer):
	tri_ma = torch.zeros([npool_vr, npool_vr], device=device)
	for ipool in range(1, npool):  # Changed so that the first pool (coarse woody debris) is all zeroes
		tri_ma[(ipool*n_soil_layer):((ipool+1)*n_soil_layer), (ipool*n_soil_layer):((ipool+1)*n_soil_layer)] = tri_ma_middle
	# end for ipool in range(npool):
	return tri_ma
# end  def tri_matrix_gas()


def tri_matrix_alternative_vectorized(nbedrock, slope, intercept, intercept_leach, device):
	# Use torch.diag with offset
	# slope = -1.2
	# intercept = -4
	rate_to_atmos = -0. # # at the surface, part of the CO2 should be released to atmos
	transport_rate_float = -10**(intercept + slope*torch.log10(zsoi[0:20]*100)) # convert zsoi from m to cm
	transport_rate_float[nbedrock:] = -10**(-30)
	transport_rate_leach = -10**(intercept_leach + slope*torch.log10(zsoi[0:20]*100)) # convert zsoi from m to cm
	transport_rate_leach[nbedrock:] = -10**(-30)

	# Create a tridiagonal matrix for each pool type
	tri_ma_middle = torch.zeros(n_soil_layer, n_soil_layer, device=device)
	tri_ma_middle = torch.diagonal_scatter(tri_ma_middle, -1*(transport_rate_float[0:n_soil_layer]+transport_rate_leach[0:n_soil_layer]), offset=0)
	tri_ma_middle = torch.diagonal_scatter(tri_ma_middle, transport_rate_float[1:n_soil_layer], offset=1)  # 1 above the main diagonal
	tri_ma_middle = torch.diagonal_scatter(tri_ma_middle, transport_rate_leach[0:(n_soil_layer-1)], offset=-1)  # 1 below the main diagonal
	tri_ma_middle[0, 0] = -1*(transport_rate_leach[0] + rate_to_atmos)
	tri_ma_middle[n_soil_layer-1, n_soil_layer-1] = -1*transport_rate_float[n_soil_layer-1]
	zero_matrix = torch.zeros(n_soil_layer, n_soil_layer, device=device)

	tri_ma = torch.block_diag(zero_matrix, tri_ma_middle, tri_ma_middle, tri_ma_middle,
				              tri_ma_middle, tri_ma_middle, tri_ma_middle)  # First is zero matrix since CWD vertical mixing is not allowed
	
	return tri_ma

# Import the improved code
def tri_matrix_old_improved(nbedrock, altmax, altmax_lastyear, som_diffus, som_adv_flux, cryoturb_diffusion_k):
	
	device = nbedrock.device

	nlevdecomp = n_soil_layer
	epsilon = 1e-30

	# change the unit from m2/yr to m2/day
	som_diffus_day = som_diffus / days_per_year
	som_adv_flux_day = som_adv_flux / days_per_year # float does not require grad
	cryoturb_diffusion_k_day = cryoturb_diffusion_k / days_per_year
	# print("som_diffus_day.requires_grad: ", som_diffus_day.requires_grad)

	# print("cryoturb_diffusion_k_day.requires_grad: ", cryoturb_diffusion_k_day.requires_grad)

	tri_ma = (torch.zeros([npool_vr, npool_vr])).to(device)

	som_adv_coef = torch.zeros(nlevdecomp+1).to(device) # SOM advective flux (m/day)
	som_diffus_coef = torch.zeros(nlevdecomp+1).to(device) # SOM diffusivity due to bio/cryo-turbation (m2/day)
	diffus = torch.zeros(nlevdecomp+1).to(device) # diffusivity (m2/day)  (includes spinup correction, if any)
	adv_flux = torch.zeros(nlevdecomp+1).to(device) # advective flux (m/day)  (includes spinup correction, if any)

	a_tri_e = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix
	b_tri_e = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix
	c_tri_e = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix
	r_tri_e = torch.zeros(nlevdecomp).to(device) #"r" vector for tridiagonal solution

	a_tri_dz = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix with considering the depth
	b_tri_dz = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix with considering the depth
	c_tri_dz = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix with considering the depth

	# d_p1_zp1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for next j
	# # (set to zero for no diffusion)
	# d_m1_zm1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for previous j
	# # (set to zero for no diffusion)
	f_p1 = torch.zeros(nlevdecomp+1).to(device) # water flux for next j
	f_m1 = torch.zeros(nlevdecomp+1).to(device) # water flux for previous j
	pe_p1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for next j
	pe_m1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for previous j
	
	w_p1 = torch.zeros(nlevdecomp+1).to(device)
	w_m1 = torch.zeros(nlevdecomp+1).to(device)

	#------ first get diffusivity / advection terms -------
	# Convert conditions to tensor operations
	active_layer_depth = torch.tensor(max(altmax.item(), altmax_lastyear.item())).to(device)
	# is_active_layer = zisoi[:nbedrock+1] < active_layer_depth
	# is_below_active_layer_and_cryoturb = (zisoi[:nbedrock+1] >= active_layer_depth) & (zisoi[:nbedrock+1] <= torch.min(torch.tensor(max_depth_cryoturb), zisoi[nbedrock+1]))
	is_active_layer = zisoi[:nlevdecomp+1] < active_layer_depth
	is_below_active_layer_and_cryoturb = (zisoi[:nlevdecomp+1] >= active_layer_depth) & (zisoi[:nlevdecomp+1] <= torch.min(torch.tensor(max_depth_cryoturb), zisoi[nlevdecomp+1]))
	is_bedrock_layer = torch.arange(nlevdecomp+1).to(device) > nbedrock
	# Fill is_active_layer with False for bedrock layers to shape = nlevdecomp+1
	# is_active_layer = torch.cat((is_active_layer, torch.full((nlevdecomp-nbedrock,), False).to(device)))
	# is_below_active_layer_and_cryoturb = torch.cat((is_below_active_layer_and_cryoturb, torch.full((nlevdecomp-nbedrock,), False).to(device)))
	# is_bedrock_layer = torch.cat((is_bedrock_layer, torch.full((nlevdecomp-nbedrock,), True).to(device)))

	# Initialize coefficients with zeros
	som_diffus_coef.fill_(0.)
	som_adv_coef.fill_(0.)

	if active_layer_depth <= max_altdepth_cryoturbation and active_layer_depth > 0.:
		# Active layer conditions
		# print("nbedrock: ", nbedrock)
		# print("Shape of som_diffus_coef: ", som_diffus_coef.shape)
		# print("Shape of active_layer_depth: ", active_layer_depth.shape)
		# print("Shape of zisoi[:nbedrock+1]: ", zisoi[:nbedrock+1].shape)
		# print("Shape of is_active_layer: ", is_active_layer.shape)
		# print("is_active_layer: ", is_active_layer)
		# print("Shape of is_below_active_layer_and_cryoturb: ", is_below_active_layer_and_cryoturb.shape)
		# print("is_below_active_layer_and_cryoturb: ", is_below_active_layer_and_cryoturb)
		# print("Shape of is_bedrock_layer: ", is_bedrock_layer.shape)
		# print("is_bedrock_layer: ", is_bedrock_layer)
		# print("Shape of cryoturb_diffusion_k_day: ", cryoturb_diffusion_k_day.shape)
		som_diffus_coef[is_active_layer] = cryoturb_diffusion_k_day
		linear_decrease_factor = (1. - (zisoi[:nlevdecomp+1][is_below_active_layer_and_cryoturb] - active_layer_depth) / (torch.min(torch.tensor(max_depth_cryoturb), zisoi[nlevdecomp+1]) - active_layer_depth))
		# print("Shape of linear_decrease_factor: ", linear_decrease_factor.shape)
		# print("Shape of som_diffus_coef: ", som_diffus_coef.shape)
		# print("Shape of is_below_active_layer_and_cryoturb: ", is_below_active_layer_and_cryoturb.shape)
		# print(torch.maximum(cryoturb_diffusion_k * linear_decrease_factor, torch.tensor(0.)).shape)
		# print("Shape of som_diffus_coef[is_below_active_layer_and_cryoturb]: ", som_diffus_coef[is_below_active_layer_and_cryoturb].shape)
		som_diffus_coef[is_below_active_layer_and_cryoturb] = torch.maximum(cryoturb_diffusion_k_day * linear_decrease_factor, torch.tensor(0.))
	elif active_layer_depth > 0.:
		# Constant advection and diffusion up to bedrock
		som_adv_coef[:nbedrock+1] = som_adv_flux_day
		som_diffus_coef[:nbedrock+1] = som_diffus_day
	# No else clause needed for completely frozen soils since the initialization already sets coefficients to 0

	# Apply mask for bedrock layers (no advection or diffusion)
	som_adv_coef[is_bedrock_layer] = 0.
	som_diffus_coef[is_bedrock_layer] = 0.


	# Initial setup - vectorized
	adv_flux = torch.where(torch.abs(som_adv_coef) < epsilon, torch.full_like(som_adv_coef, epsilon), som_adv_coef)
	diffus = torch.where(torch.abs(som_diffus_coef) < epsilon, torch.full_like(som_diffus_coef, epsilon), som_diffus_coef)
	# print("diffus", diffus)
	# print("adv_flux requires grad", adv_flux.requires_grad)
	# print("diffus requires grad", diffus.requires_grad)

	# Initialize tensors
	f_m1 = adv_flux.clone()
	f_p1 = torch.cat((adv_flux[1:], torch.tensor([0.]).to(device)))  # Shift adv_flux down and pad with 0
	w_m1 = torch.zeros_like(adv_flux)
	w_p1 = torch.zeros_like(adv_flux)
	d_m1_zm1 = torch.zeros_like(adv_flux)
	d_p1_zp1 = torch.zeros_like(adv_flux)

	# Calculations that apply for all layers except the special cases at the top and bottom
	# w_m1[1:] = (zisoi[:nlevdecomp+1][:-1] - zsoi[:nlevdecomp+1][:-1]) / dz_node[:nlevdecomp+1][1:]  # Skip the first layer for w_m1
	# Try to avoid inplace operations
	w_m1 = torch.cat([w_m1[:1], (zisoi[:nlevdecomp+1][:-1] - zsoi[:nlevdecomp+1][:-1]) / dz_node[:nlevdecomp+1][1:]])
	# At the bottom, assume no gradient in dz (i.e., they're the same)
	# w_p1[:-2] = (zsoi[:nlevdecomp+1][1:-1] - zisoi[:nlevdecomp+1][:-2]) / dz_node[:nlevdecomp+1][1:-1]  # Skip the last two layers for w_p1
	# Try to avoid inplace operations
	w_p1 = torch.cat([(zsoi[:nlevdecomp+1][1:-1] - zisoi[:nlevdecomp+1][:-2]) / dz_node[:nlevdecomp+1][1:-1], w_p1[-2:]])
	

	# Harmonic mean for internal layers, with adjustments for boundary conditions
	inner_layers = (diffus[1:] > 0) & (diffus[:-1] > 0)  # Boolean mask for layers where both adjacent diffusivities are > 0
	# d_m1_zm1[1:] = torch.where(inner_layers, 1. / ((1. - w_m1[1:]) / diffus[1:] + w_m1[1:] / diffus[:-1]), 0.)
	# d_p1_zp1[:-1] = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_m1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])
	d_m1_zm1 = torch.cat([d_m1_zm1[:1], torch.where(inner_layers, 1. / ((1. - w_m1[1:]) / diffus[1:] + w_m1[1:] / diffus[:-1]), torch.zeros_like(d_m1_zm1[1:]))])
	d_p1_zp1_temp = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_m1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])
	d_p1_zp1 = torch.cat([d_p1_zp1_temp, d_p1_zp1[-1:]])

	# Adjust for dz_node scaling
	d_m1_zm1[1:] /= dz_node[:nlevdecomp+1][1:]
	d_p1_zp1[:-1] /= dz_node[:nlevdecomp+1][1:]

	# print("d_p1_zp1", d_p1_zp1)


	# Layer lower than nbedrock-1: d_p1_zp1 = d_m1_zm1
	

	# Bottom layer - assume no gradient in dz
	# w_m1[-1] = (zisoi[:nlevdecomp+1][-2] - zsoi[:nlevdecomp+1][-2]) / dz_node[:nlevdecomp+1][-1]
	# d_m1_zm1[-1] = 1. / ((1. - w_m1[-1]) / diffus[-1] + w_m1[-1] / diffus[-2]) if diffus[-1] > 0 and diffus[-2] > 0 else 0.
	# d_m1_zm1[-1] /= dz_node[:nlevdecomp+1][-1]
	w_m1_bottom = (zisoi[:nlevdecomp+1][-2] - zsoi[:nlevdecomp+1][-2]) / dz_node[:nlevdecomp+1][-1]
	d_m1_zm1_bottom = 1. / ((1. - w_m1_bottom) / diffus[-1] + w_m1_bottom / diffus[-2]) if diffus[-1] > 0 and diffus[-2] > 0 else 0.
	d_m1_zm1_bottom /= dz_node[:nlevdecomp+1][-1]

	d_m1_zm1= torch.cat([d_m1_zm1[:-1], d_m1_zm1_bottom.unsqueeze(0)])
	d_p1_zp1 = torch.cat([d_p1_zp1[:nbedrock], d_m1_zm1[nbedrock:]])


	# d_p1_zp1[nbedrock-1:] = d_m1_zm1[nbedrock-1:]
	# # No advective flux for the layer between nbedrock and nlevdecomp
	# f_p1[nbedrock-1:] = 0

	# No advective flux for the layer between nbedrock and nlevdecomp without in-place modification
	f_p1 = torch.cat([f_p1[:nbedrock], torch.zeros_like(f_p1[nbedrock:])])

	# assign 0 if nlevdecomp > nbedrock
	# for example, if nbedrock = 7, only select top 6 elements
	w_p1 = torch.where(torch.arange(nlevdecomp+1).to(device) > nbedrock-1, torch.zeros_like(w_p1), w_p1)

	# Boundary conditions
	# Top layer
	w_p1[0] = (zsoi[:nlevdecomp+1][1] - zisoi[:nlevdecomp+1][0]) / dz_node[:nlevdecomp+1][1]
	d_p1_zp1[0] = 1. / ((1. - w_p1[0]) / diffus[0] + w_p1[0] / diffus[1]) if diffus[1] > 0 and diffus[0] > 0 else 0.
	d_p1_zp1[0] /= dz_node[:nlevdecomp+1][1]

	# print((zsoi[:nlevdecomp+1][1] - zisoi[:nlevdecomp+1][0]) / dz_node[:nlevdecomp+1][1])

	# print("w_p1", w_p1)
	# print("w_m1", w_m1)
	# # print(1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]))
	# print("d_p1_zp1", d_p1_zp1)
	# print("d_m1_zm1", d_m1_zm1)




	# Peclet numbers
	pe_m1 = torch.where(d_m1_zm1 == 0, torch.zeros_like(f_m1), f_m1 / (d_m1_zm1 + epsilon))
	pe_p1 = torch.where(d_p1_zp1 == 0, torch.zeros_like(f_p1), f_p1 / (d_p1_zp1 + epsilon))

	# Pre-compute the 'aaa' values for Patankar functions
	aaa_m = torch.maximum(torch.zeros_like(pe_m1), (1. - 0.1 * pe_m1.abs())**5)
	aaa_p = torch.maximum(torch.zeros_like(pe_p1), (1. - 0.1 * pe_p1.abs())**5)

	# Vectorized computation of tridiagonal coefficients
	a_tri_e = -(d_m1_zm1 * aaa_m + torch.maximum(f_m1, torch.zeros_like(f_m1)))
	c_tri_e = -(d_p1_zp1 * aaa_p + torch.maximum(-f_p1, torch.zeros_like(f_p1)))
	b_tri_e = -a_tri_e - c_tri_e

	# print("aaa_p", aaa_p)
	# print("f_p1", f_p1)

	# print("Shape of d_m1_zm1: ", d_m1_zm1.shape)
	# print("Shape of pe_m1: ", pe_m1.shape)
	# print("Shape of pe_p1: ", pe_p1.shape)
	# print("Shape of aaa_m: ", aaa_m.shape)
	# print("Shape of f_m1: ", f_m1.shape)



	a_tri_dz = a_tri_e / dz[:nlevdecomp+1]
	b_tri_dz = b_tri_e / dz[:nlevdecomp+1]
	c_tri_dz = c_tri_e / dz[:nlevdecomp+1]
	a_tri_dz = a_tri_dz[:-1]
	b_tri_dz = b_tri_dz[:-1]
	c_tri_dz = c_tri_dz[:-1]

	# # no vertical transportation in CWD
	# for i in range(1, npool): #= 2 : npool
	# 	for j in range(nlevdecomp): #= 1 : nlevdecomp
	# 		tri_ma[j+(i)*nlevdecomp,j+(i)*nlevdecomp] = b_tri_dz[j]
	# 		if j == 0:   # upper boundary
	# 			tri_ma[j+(i)*nlevdecomp,j+(i)*nlevdecomp] = -c_tri_dz[j]
	# 			# print(i, j, 1+(i)*nlevdecomp, tri_ma[1+(i)*nlevdecomp,1+(i)*nlevdecomp])

	# 		# end if j == 0: 
	# 		if j == (nlevdecomp-1):  # bottom boundary
	# 			tri_ma[nlevdecomp-1+(i)*nlevdecomp,nlevdecomp-1+(i)*nlevdecomp] = -a_tri_dz[nlevdecomp-1]
	# 		# end if j == (nlevdecomp-1):
	# 		if j < (nlevdecomp-1): # avoid tranfer from for example, litr3_20th layer to soil1_1st layer
	# 			tri_ma[j+(i)*nlevdecomp,j+1+(i)*nlevdecomp] = c_tri_dz[j]

	# 		# end if j < (nlevdecomp-1): 

		
	# 		if j > 0: # avoid tranfer from for example,soil1_1st layer to litr3_20th layer
	# 			tri_ma[j+(i)*nlevdecomp,j-1+(i)*nlevdecomp] = a_tri_dz[j]

	# for i in range(1, npool):
	# 	idx = torch.arange(nlevdecomp) + i * nlevdecomp
	# 	tri_ma[idx, idx] = b_tri_dz

	# 	# Upper boundary
	# 	tri_ma[idx[0], idx[0]] = -c_tri_dz[0]

	# 	# Bottom boundary
	# 	tri_ma[idx[-1], idx[-1]] = -a_tri_dz[-1]

	# 	# Off-diagonals
	# 	idx = torch.arange(nlevdecomp-1) + i * nlevdecomp
	# 	tri_ma[idx, idx+1] = c_tri_dz[:-1]
	# 	tri_ma[idx+1, idx] = a_tri_dz[1:]

	# Try to get rid of the for loop
	# Expand a_tri_dz, b_tri_dz, c_tri_dz
	expanded_a = torch.cat([torch.zeros(20, device=device), a_tri_dz.repeat(npool-1)])
	expanded_b = torch.cat([torch.zeros(20, device=device), b_tri_dz.repeat(npool-1)])
	expanded_c = torch.cat([torch.zeros(20, device=device), c_tri_dz.repeat(npool-1)])

	# Fill diagonal with expanded_b
	tri_ma.fill_diagonal_(0)  # Ensure diagonal is clear before setting
	tri_ma[torch.arange(140), torch.arange(140)] = expanded_b

	# Adjust for upper boundary conditions across blocks
	upper_boundary_indices = torch.arange(20, 140, 20)
	tri_ma[upper_boundary_indices, upper_boundary_indices] = -expanded_c[upper_boundary_indices]

	# Adjust for bottom boundary conditions across blocks
	bottom_boundary_indices = torch.arange(39, 140, 20)
	tri_ma[bottom_boundary_indices, bottom_boundary_indices] = -expanded_a[bottom_boundary_indices]

	# Fill off-diagonals for c_tri_dz
	off_diag_indices_right = torch.arange(19, 139)  # Right off-diagonal
	tri_ma[torch.arange(19, 139), torch.arange(20, 140)] = expanded_c[off_diag_indices_right]
	

	# Fill off-diagonals for a_tri_dz
	off_diag_indices_left = torch.arange(20, 140)  # Left off-diagonal
	tri_ma[torch.arange(20, 140), torch.arange(19, 139)] = expanded_a[off_diag_indices_left]

	# Adjust for upper boundary conditions across blocks
	tri_ma[upper_boundary_indices, upper_boundary_indices-1] = 0
	
	# Adjust for bottom boundary conditions across blocks
	bottom_boundary_indices = torch.arange(39, 120, 20)
	tri_ma[bottom_boundary_indices, bottom_boundary_indices+1] = 0




	# # no vertical transportation in CWD
	# for i in range(1, npool): #= 2 : npool
	# 	for j in range(nlevdecomp): #= 1 : nlevdecomp
	# 		tri_ma[j+(i)*nlevdecomp,j+(i)*nlevdecomp] = b_tri_dz[j]
	# 		if j == 0:   # upper boundary
	# 			tri_ma[1+(i)*nlevdecomp,1+(i)*nlevdecomp] = -c_tri_dz[1]
	# 			# print(i, j, 1+(i)*nlevdecomp, tri_ma[1+(i)*nlevdecomp,1+(i)*nlevdecomp])

	# 		# end if j == 0: 
	# 		if j == (nlevdecomp-1):  # bottom boundary
	# 			tri_ma[nlevdecomp-1+(i)*nlevdecomp,nlevdecomp-1+(i)*nlevdecomp] = -a_tri_dz[nlevdecomp-1]
	# 		# end if j == (nlevdecomp-1):
	# 		if j < (nlevdecomp-1): # avoid tranfer from for example, litr3_20th layer to soil1_1st layer
	# 			tri_ma[j+(i)*nlevdecomp,j+1+(i)*nlevdecomp] = c_tri_dz[j]

	# 		# end if j < (nlevdecomp-1): 

		
	# 		if j > 0: # avoid tranfer from for example,soil1_1st layer to litr3_20th layer
	# 			tri_ma[j+(i)*nlevdecomp,j-1+(i)*nlevdecomp] = a_tri_dz[j]

	

	return tri_ma


def tri_matrix_old(nbedrock, altmax, altmax_lastyear, som_diffus, som_adv_flux, cryoturb_diffusion_k):
	device = nbedrock.device

	nlevdecomp = n_soil_layer
	epsilon = 1e-30

	# change the unit from m2/yr to m2/day, in consistant with the unit of input
	som_diffus = som_diffus/days_per_year
	som_adv_flux = som_adv_flux/days_per_year
	cryoturb_diffusion_k = cryoturb_diffusion_k/days_per_year

	tri_ma = (torch.zeros([npool_vr, npool_vr])).to(device)

	som_adv_coef = torch.zeros(nlevdecomp+1).to(device) # SOM advective flux (m/day)
	som_diffus_coef = torch.zeros(nlevdecomp+1).to(device) # SOM diffusivity due to bio/cryo-turbation (m2/day)
	diffus = torch.zeros(nlevdecomp+1).to(device) # diffusivity (m2/day)  (includes spinup correction, if any)
	adv_flux = torch.zeros(nlevdecomp+1).to(device) # advective flux (m/day)  (includes spinup correction, if any)

	a_tri_e = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix
	b_tri_e = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix
	c_tri_e = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix
	r_tri_e = torch.zeros(nlevdecomp).to(device) #"r" vector for tridiagonal solution

	a_tri_dz = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix with considering the depth
	b_tri_dz = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix with considering the depth
	c_tri_dz = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix with considering the depth

	d_p1_zp1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for next j
	# (set to zero for no diffusion)
	d_m1_zm1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for previous j
	# (set to zero for no diffusion)
	f_p1 = torch.zeros(nlevdecomp+1).to(device) # water flux for next j
	f_m1 = torch.zeros(nlevdecomp+1).to(device) # water flux for previous j
	pe_p1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for next j
	pe_m1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for previous j
	
	w_p1 = torch.zeros(nlevdecomp+1).to(device)
	w_m1 = torch.zeros(nlevdecomp+1).to(device)
	#------ first get diffusivity / advection terms -------
	# use different mixing rates for bioturbation and cryoturbation, with fixed bioturbation and cryoturbation set to a maximum depth
	if  max(altmax, altmax_lastyear) <= max_altdepth_cryoturbation and max(altmax, altmax_lastyear) > 0.:
		# use mixing profile modified slightly from Koven et al. (2009): constant through active layer, linear decrease from base of active layer to zero at a fixed depth
		for j in range(nlevdecomp+1):
			if j <= nbedrock:
				if zisoi[j] < max(altmax, altmax_lastyear):
					som_diffus_coef[j] = cryoturb_diffusion_k
					som_adv_coef[j] = 0.
				else:
					som_diffus_coef[j] = max(cryoturb_diffusion_k*(1.-(zisoi[j]-max(altmax, altmax_lastyear))/(min(max_depth_cryoturb, zisoi[nbedrock+1]) - max(altmax, altmax_lastyear))), 0.) # go linearly to zero between ALT and max_depth_cryoturb
					som_adv_coef[j] = 0.
				# end if zisoi[j] < max(altmax, altmax_lastyear):
			else:
				som_adv_coef[j] = 0.
				som_diffus_coef[j] = 0.
			# end if j <= nbedrock:
		# end for j in range(nlevdecomp+1):
	elif max(altmax, altmax_lastyear) > 0.:
		# constant advection, constant diffusion
		for j in range(nlevdecomp+1):
			if j <= nbedrock:
				som_adv_coef[j] = som_adv_flux
				som_diffus_coef[j] = som_diffus
			else:
				som_adv_coef[j] = 0.
				som_diffus_coef[j] = 0.
			# end if j <= nbedrock:
		# end for j in range(nlevdecomp+1):
	else:
		# completely frozen soils--no mixing
		for j in range(nlevdecomp+1):
			som_adv_coef[j] = 0.
			som_diffus_coef[j] = 0.
		# end for j in range(nlevdecomp+1):
	# end if  max(altmax, altmax_lastyear) <= max_altdepth_cryoturbation and max(altmax, altmax_lastyear) > 0.:

	# for j in torch.arange(nlevdecomp+1):
	#	if abs(som_adv_coef[j])  < epsilon:
	#		adv_flux[j] = epsilon
	#	else:
	#		adv_flux[j] = som_adv_coef[j]
	#	# end if abs(som_adv_coef[j])  < epsilon:
	#	if abs(som_diffus_coef[j])  < epsilon:
	#		diffus[j] = epsilon
	#	else:
	#		diffus[j] = som_diffus_coef[j]
	#	# end abs(som_diffus_coef[j])  < epsilon:
	## end for j in range(nlevdecomp+1):
	
	# Set Pe (Peclet #) and D/dz throughout column
	for j in range(nlevdecomp+1):
		if abs(som_adv_coef[j])  < epsilon:
			adv_flux[j] = epsilon
		else:
			adv_flux[j] = som_adv_coef[j]
		# end if abs(som_adv_coef[j])  < epsilon:
		
		if abs(som_diffus_coef[j])  < epsilon:
			diffus[j] = epsilon
		else:
			diffus[j] = som_diffus_coef[j]
		# end if abs(som_diffus_coef[j])  < epsilon:
		
		
		# Calculate the D and F terms in the Patankar algorithm
		if j == 0:
			d_m1_zm1[j] = 0.
			w_m1[j] = 0.
			w_p1[j] = (zsoi[j+1] - zisoi[j]) / dz_node[j+1]
			
			if diffus[j+1] > 0. and diffus[j] > 0.:
				d_p1_zp1[j] = 1. / ((1. - w_p1[j].clone()) / diffus[j].clone() + w_p1[j].clone() / diffus[j+1].clone()) # Harmonic mean of diffus
			else:
				d_p1_zp1[j] = 0.
			# end diffus[j+1] > 0. and diffus[j] > 0.:
			
			d_p1_zp1[j] = d_p1_zp1[j] / dz_node[j+1]
			f_m1[j] = adv_flux[j]  # Include infiltration here
			f_p1[j] = adv_flux[j+1]
			# pe_m1[j] = 0.
			# pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j] # Peclet #
		elif j >= nbedrock-1:
			# At the bottom, assume no gradient in d_z (i.e., they're the same)
			w_m1[j] = (zisoi[j-1] - zsoi[j-1]) / dz_node[j]
			w_p1[j] = 0.

			if diffus[j] > 0. and diffus[j-1] > 0.:
				d_m1_zm1[j] = 1. / ((1. - w_m1[j].clone()) / diffus[j].clone() + w_m1[j].clone() / diffus[j-1].clone()) # Harmonic mean of diffus
			else:
				d_m1_zm1[j] = 0.
			# end diffus[j] > 0. and diffus[j-1] > 0.:
			
			d_m1_zm1[j] = d_m1_zm1[j] / dz_node[j]
			d_p1_zp1[j] = d_m1_zm1[j] # Set to be the same
			f_m1[j] = adv_flux[j]
			# f_p1(j) = adv_flux(j+1)
			f_p1[j] = 0.
			# pe_m1[j] = f_m1[j].clone() / d_m1_zm1[j] # Peclet #
			# pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j] # Peclet #
		else:
			# Use distance from j-1 node to interface with j divided by distance between nodes
			w_m1[j] = (zisoi[j-1] - zsoi[j-1]) / dz_node[j]
			w_p1[j] = (zsoi[j+1] - zisoi[j]) / dz_node[j+1]
			
			if diffus[j-1] > 0. and diffus[j] > 0.:
				d_m1_zm1[j] = 1. / ((1. - w_m1[j].clone()) / diffus[j].clone() + w_m1[j].clone() / diffus[j-1].clone()) # Harmonic mean of diffus
			else:
				d_m1_zm1[j] = 0.
			# end diffus[j-1] > 0. and diffus[j] > 0.:

			if diffus[j+1] > 0. and diffus[j] > 0.:
				d_p1_zp1[j] = 1. / ((1. - w_p1[j].clone()) / diffus[j].clone() + w_p1[j].clone() / diffus[j+1].clone()) # Harmonic mean of diffus
			else:
				d_p1_zp1[j] = (1. - w_m1[j].clone()) * diffus[j].clone() + w_p1[j].clone() * diffus[j+1].clone() # Arithmetic mean of diffus
			# end if diffus[j+1] > 0. and diffus[j] > 0.:
			
			d_m1_zm1[j] = d_m1_zm1[j] / dz_node[j]
			d_p1_zp1[j] = d_p1_zp1[j] / dz_node[j+1]
			f_m1[j] = adv_flux[j]
			f_p1[j] = adv_flux[j+1]
			# pe_m1[j] = f_m1[j].clone() / d_m1_zm1[j] # Peclet #
			# pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j] # Peclet #
		# end j == 0:
	# end for j in range(nlevdecomp+1): 
	
	# Peclet #
	for j in range(nlevdecomp+1):
		if d_m1_zm1[j] == 0.:
			pe_m1[j] = 0.
		else:
			pe_m1[j] = f_m1[j].clone() / d_m1_zm1[j]
		#end if d_m1_zm1[j] == 0.:
		if d_p1_zp1[j] == 0.:
			pe_p1[j] = 0.
		else:
			pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j]
	# end for j in torch.arange(nlevdecomp+1):

	# Calculate the tridiagonal coefficients
	for j in range(-1, (nlevdecomp+1)): # 0:nlevdecomp+1
		if j == -1: # top layer (atmosphere)
			pass
			# a_tri(j) = 0.
			# b_tri(j) = 1.
			# c_tri(j) = -1.
			# b_tri_e(j) = b_tri(j)
		elif j == 0: #  Set statement functions
			aaa = max (0., (1. - 0.1 * abs(pe_m1[j]))**5)  # A function from Patankar, Table 5.2, pg 95
			a_tri_e[j] = -(d_m1_zm1[j] * aaa + max( f_m1[j], 0.)) # Eqn 5.47 Patankar
			aaa = max (0., (1. - 0.1 * abs(pe_p1[j]))**5)  # A function from Patankar, Table 5.2, pg 95
			c_tri_e[j] = -(d_p1_zp1[j] * aaa + max(-f_p1[j], 0.))
			b_tri_e[j] = -a_tri_e[j] - c_tri_e[j]
		elif j < nlevdecomp:
			aaa = max (0., (1. - 0.1 * abs(pe_m1[j]))**5) # A function from Patankar, Table 5.2, pg 95
			a_tri_e[j] = -(d_m1_zm1[j] * aaa + max( f_m1[j], 0.)) # Eqn 5.47 Patankar
			aaa = max (0., (1. - 0.1 * abs(pe_p1[j]))**5)  # A function from Patankar, Table 5.2, pg 95
			c_tri_e[j] = -(d_p1_zp1[j] * aaa + max(-f_p1[j], 0.))
			b_tri_e[j] = -a_tri_e[j] - c_tri_e[j]
		else: # j==nlevdecomp; 0 concentration gradient at bottom
			pass
			# a_tri(j) = -1.
			# b_tri(j) = 1.
			# c_tri(j) = 0.
		# end if j == 0: 
	# end for j in range(nlevdecomp+2)
	
	for j in range(nlevdecomp):
		# elements for vertical matrix match unit g/m3,  Tri/dz
		a_tri_dz[j] = a_tri_e[j] / dz[j]
		b_tri_dz[j] = b_tri_e[j] / dz[j]
		c_tri_dz[j] = c_tri_e[j] / dz[j]
	# end for j in range(nlevdecomp):
	

	for i in range(1, npool):
		start_idx = i * nlevdecomp
		end_idx = (i + 1) * nlevdecomp

		# Main diagonal
		diag_indices = torch.arange(start_idx, end_idx)
		tri_ma[diag_indices, diag_indices] = b_tri_dz

		# Upper boundary condition adjustment
		if nlevdecomp > 1:
			tri_ma[start_idx, start_idx] = -c_tri_dz[1]


		# Bottom boundary condition adjustment
		tri_ma[end_idx - 1, end_idx - 1] = -a_tri_dz[-1]

		# Upper and lower diagonals
		# Ensure indices don't go out of bounds
		if nlevdecomp > 2:  # Only proceed if there are at least 3 layers, allowing for upper and lower diagonals
			upper_diag_indices = diag_indices[:-1] + 1
			lower_diag_indices = diag_indices[1:] - 1

			tri_ma[diag_indices[:-1], upper_diag_indices] = c_tri_dz[:-1]
			tri_ma[diag_indices[1:], lower_diag_indices] = a_tri_dz[1:]
	
	return tri_ma

# end def tri_matrix

def catanf(t1):
	catanf_results = 11.75 +(29.7 / np.pi) * torch.arctan( torch.tensor(np.pi) * 0.031  * ( t1 - 15.4 ))
	return catanf_results
# end def catanf



