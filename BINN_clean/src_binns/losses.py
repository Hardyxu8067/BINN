import torch
import visualization_utils


def binns_loss(y_pred, y_true, pred_para, plot_path=""):
	"""
	Computes the main losses used in BINN.
	 
	y_pred: predicted SOC, shape [batch, observations_per_site]
	y_true: true_SOC, shape [batch, observations_per_site]
	Many entries can be nan (nan positions should be the same for y_pred and y_true).
	pred_para: predicted parameters, shape [batch, num_params]

	Returns
	1) L1 loss
	2) Smooth L1 loss
	3) L2 loss
	4) Parameter regularization loss (penalizes extreme param values)
	5) Modeling inefficiency (1 - NSE, or 1 - R^2)

	All of these (except parameter regularization loss) compare how close
	predictions (y_pred) are with observations (y_true).
	"""
	# Predicted (simulated) SOC
	soc_simu = y_pred

	# Observed SOC
	soc_true = y_true

	# Predicted parameters
	pred_para = pred_para

	# flatten simulated and true SOC
	soc_simu_vector = torch.reshape(soc_simu, [1, -1])
	soc_true_vector = torch.reshape(soc_true, [1, -1])

	# exclude nan
	valid_loc = torch.where(torch.isnan(soc_simu_vector+soc_true_vector) == False)
	soc_simu_vector = soc_simu_vector[valid_loc]
	soc_true_vector = soc_true_vector[valid_loc]

	# If desired, plot true vs predicted here
	if plot_path != "":
		visualization_utils.plot_true_vs_predicted(plot_path, soc_simu_vector, soc_true_vector)

	# modeling inefficiency
	modeling_inefficiency = torch.sum((soc_simu_vector - soc_true_vector)**2)/torch.sum((soc_true_vector - torch.mean(soc_true_vector))**2)

	# Regularization for predicted parameters using cosh
	# Encourage parameters to be around 0.5
	target_value = 0.5
	scale_factor = 10
	param_reg_loss = torch.mean(torch.cosh(scale_factor*(pred_para - target_value)) - 1)

	# Calculate the supervised losses
	l1_loss = torch.nn.functional.l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	smooth_l1_loss = torch.nn.functional.smooth_l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	l2_loss = torch.nn.functional.mse_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	return l1_loss, smooth_l1_loss, l2_loss, param_reg_loss, modeling_inefficiency
# end binns loss

