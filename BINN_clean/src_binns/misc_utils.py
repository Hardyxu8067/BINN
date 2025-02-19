import collections
import torch
import torch.nn
import numpy as np


@torch.no_grad()
def update_bn_custom(loader, model, device=None):
	r"""Method from SWA library, but adapted to use our custom data,
	where each dataloader batch contains x,z,y,profile_id and x,z
	need to be passed through the model.

	Documentation from SWA:
	Updates BatchNorm running_mean, running_var buffers in the model.
	It performs one pass over data in `loader` to estimate the activation
	statistics for BatchNorm layers in the model.
	Args:
		loader (torch.utils.data.DataLoader): dataset loader to compute the
			activation statistics on. Each data batch should be either a
			tensor, or a list/tuple whose first element is a tensor
			containing data.
		model (torch.nn.Module): model for which we seek to update BatchNorm
			statistics.
		device (torch.device, optional): If set, data will be transferred to
			:attr:`device` before being passed into :attr:`model`.

	Example:
		>>> # xdoctest: +SKIP("Undefined variables")
		>>> loader, model = ...
		>>> torch.optim.swa_utils.update_bn(loader, model)

	.. note::
		The `update_bn` utility assumes that each data batch in :attr:`loader`
		is either a tensor or a list or tuple of tensors; in the latter case it
		is assumed that :meth:`model.forward()` should be called on the first
		element of the list or tuple corresponding to the data batch.
	"""
	momenta = {}
	for module in model.modules():
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.reset_running_stats()
			momenta[module] = module.momentum

	if not momenta:
		return

	was_training = model.training
	model.train()
	for module in momenta.keys():
		module.momentum = None

	for input in loader:
		batch_x, batch_y, batch_z, batch_profile_id = input
		if device is not None:
			batch_x = batch_x.to(device)
			batch_z = batch_z.to(device)

		model(batch_x, batch_z, whether_predict=0)

	for bn_module in momenta.keys():
		bn_module.momentum = momenta[bn_module]
	model.train(was_training)



def zero_gradients(x):
	"""
	This was formerly an old function in "torch.autograd.gradcheck".
	It is now removed so we reproduce the code here, see
	https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462
	"""
	if isinstance(x, torch.Tensor):
		if x.grad is not None:
			x.grad.detach_()
			x.grad.zero_()
	elif isinstance(x, collections.abc.Iterable):
		for elem in x:
			zero_gradients(elem)

def _find_z(new_input, batch_y, batch_z, model, criterion, h):
	'''
	Finding the direction in the regularizer
	'''
	batch_x.requires_grad_()
	outputs, h5, new_input = model.eval().partial(batch_x, batch_z, whether_predict=0, return_input=True)
	loss_z = criterion(outputs, batch_y)
	loss_z.backward(torch.ones(batch_y.shape).to(batch_y.device))
	grad = new_input.grad.data + 0.0
	norm_grad = grad.norm().item()
	z = torch.sign(grad).detach() + 0.
	z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
	zero_gradients(new_input)
	model.zero_grad()

	return z, norm_grad, new_input


# def regularizer(batch_x, batch_y, batch_z, model, criterion, h = 3.):
# 	'''
# 	ARCHIVED. For now, instead of using gradient of loss w.r.t. input, use gradient of params w.r.t. input.
# 	Regularizer term in CURE

# 	Returns both curvature loss
# 	'''
# 	with torch.no_grad():
# 		outputs, h5, new_input = model.eval()(batch_x, batch_z, whether_predict=0, return_input=True)

# 	# z is a direction of high curvature (pointing in similar direction to the gradient)
# 	z, norm_grad, new_input = _find_z(batch_x, batch_y, batch_z, model, criterion, h)

# 	new_input.requires_grad_()

# 	# Compute
# 	outputs_pos, _ = model.eval().partial_forward(new_input + z)
# 	outputs_orig, _ = model.eval().partial_forward(new_input)
# 	loss_pos = criterion(outputs_pos, batch_y)
# 	loss_orig = criterion(outputs_orig, batch_y)
# 	grad_diff = torch.autograd.grad((loss_pos-loss_orig), new_input,
# 									 grad_outputs=torch.ones(batch_y.shape).to(batch_y.device),
# 									 create_graph=True)[0]
# 	reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
# 	model.zero_grad()

# 	return torch.sum(reg) / float(new_input.size(0)), norm_grad



def select_depth(tensor_simu, tensor_frocing_steady_state, tensor_obs_layer_depth):
	simu = tensor_simu
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state
	obs_layer_depth = tensor_obs_layer_depth
	device = tensor_simu.device

	# depth of the node
	zsoi_local = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029])
	zsoi = zsoi_local.to(device)
	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = simu.shape[0]
	simu_ouput = (torch.ones((profile_num, 200))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_simu = tensor_simu[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]
		profile_obs_layer_depth = obs_layer_depth[iprofile, :]
		valid_layer_loc = torch.where(torch.isnan(profile_obs_layer_depth) == False)[0]

		if torch.isnan(torch.sum(profile_simu)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:

			# print(profile_para)
			# model simulation
			profile_simu_soc = profile_simu

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
	return simu_ouput
# end nn_model



def inject_noise(model, noise_std):
	"""
	Adds a small amount of random noise to the parameters of the network.

	Source: https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/incremental_cifar/incremental_cifar_experiment.py
	"""
	with torch.no_grad():
		for param in model.parameters():
			param.add_(torch.randn(param.size(), device=param.device) * noise_std)



def print_summary(tensor, message="", dim=None):
	if dim is None:
		print(message, "- Shape", tensor.shape, "Mean", tensor.mean(), "Std", tensor.std(), "Min", tensor.min(), "Max", tensor.max())
	else:
		print(message, "- Shape", tensor.shape, "Mean", tensor.mean(dim=dim), "Std", tensor.std(dim=dim), "Min", tensor.min(dim=dim).values, "Max", tensor.max(dim=dim).values)


import sys

class Logger(object):
	"""
	Logger that writes printed statements to both stdout (terminal) and log file.
	To use, set "sys.stdout = Logger(log_file)"
	Source: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
	"""
	def __init__(self, log_file):
		self.terminal = sys.stdout
		self.log = open(log_file, "a")
   
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass    

