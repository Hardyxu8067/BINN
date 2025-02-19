import torch
import visualization_utils

#---------------------------------------------------
# define the loss function                          
#---------------------------------------------------
def binns_loss(y_pred, y_true, pred_para, plot_path=""):
	# process modeling
	soc_simu = y_pred
	# observations
	soc_true = y_true
	# predicted parameters
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

	# NOT USED ANYMORE: Attempts to penalize extreme values of the beta parameter.
	# # Gradient of beta w.r.t. batch_x
	# grad_beta = torch.autograd.grad(outputs=beta, inputs=model_input, grad_outputs=torch.ones_like(beta), retain_graph=True)[0]
	# # only consider the gradient that is not zero
	# non_zero_idx = torch.nonzero(grad_beta)
	# grad_beta = grad_beta[non_zero_idx[:, 0], non_zero_idx[:, 1], non_zero_idx[:, 2], non_zero_idx[:, 3]]
	# # Use torch.autograd.grad for inputs for one profile
	# for i in range(beta.shape[0]):
	# 	grad_beta_temp = torch.autograd.grad(outputs=beta[i], inputs=model_input[i, :], grad_outputs=torch.ones_like(beta[i]), retain_graph=True)[0]
	# 	if i == 0:
	# 		grad_beta = grad_beta_temp
	# 	else:
	# 		grad_beta = torch.cat((grad_beta, grad_beta_temp), dim=0)
	# grad_norm = torch.norm(grad_beta, p=2)

	# print("Beta gradient norm", grad_norm)

	# modeling_inefficiency = torch.sum((soc_simu_vector - soc_true_vector)**2)/len(soc_true_vector) 
	# lambda_reg = 0.1

	# # Calculate the penalty if beta is too large
	# penalty_threshold = 0.8
	# transformed_threshold = -0.8  # Transformed threshold for negative values
	# penalty_weight = 20  # Adjust the weight as needed

	# # Inverting beta values so that high values become negative
	# inverted_beta = -beta

	# # Applying threshold to inverted beta
	# thresholded_beta = torch.nn.Threshold(transformed_threshold, 0)(inverted_beta)

	# # Inverting back to positive values and applying penalty
	# beta_penalty = penalty_weight * (-(thresholded_beta) - penalty_threshold)**2

	
	# # Sum the penalty across the batch
	# total_beta_penalty = beta_penalty.sum()

	# ## Variance term ##
	# # Calculate variance of predicted parameters
	# var_predicted_para = torch.var(pred_para[:, 20]) # calculate variance of the 21st parameter beta
	# # var_predicted_para = torch.mean(var_predicted_para)
	# # print("Variance of predicted parameters", var_predicted_para)

	# # Normalize or scale the variance term
	# scale_factor = 1e5
	# scaled_variance = scale_factor * var_predicted_para

	# # Weighting factor for variance term
	# variance_weight = 0

	# Regularization for predicted parameters using cosh
	# Encourage parameters to be around 0.5
	target_value = 0.5
	scale_factor = 10
	param_reg_loss = torch.mean(torch.cosh(scale_factor*(pred_para - target_value)) - 1)

	# Calculate the supervised losses
	l1_loss = torch.nn.functional.smooth_l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	l2_loss = torch.nn.functional.mse_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	return l1_loss, l2_loss, param_reg_loss, modeling_inefficiency
# end binns loss


#---------------------------------------------------
# simplified loss function that only takes in pred/true
# and returns a single value (smooth l1). Used for CURE
# curvature regularization.
#---------------------------------------------------
def binns_loss_simple(y_pred, y_true):
	pred_para = torch.zeros_like(y_pred)  # Not used
	l1_loss, _, _, _ = binns_loss(y_pred, y_true, pred_para)
	return l1_loss
