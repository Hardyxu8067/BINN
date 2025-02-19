import torch
import torch.nn as nn
import torch.nn.functional as F
from lipmlp import lipmlp
from fun_matrix_clm5_vectorized import fun_model_simu, fun_model_prediction
from pe_gcn_model import GridCellSpatialRelationEncoder
from misc_utils import select_depth
from spatial_utils import *

class mlp(torch.nn.Module):
	"""
	New MLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
	"""
	def __init__(self, dims, use_bn=False, dropout_prob=0.0, leaky_relu=False, init='xavier_uniform'):
		"""
		dim[0]: input dim
		dim[1:-1]: hidden dims
		dim[-1]: out dim

		assume len(dims) >= 3
		"""
		super().__init__()

		self.layers = torch.nn.ModuleList()
		self.use_bn = use_bn
		self.dropout_prob = dropout_prob
		if use_bn:
			self.bns = torch.nn.ModuleList()
		if dropout_prob > 0:
			self.dropout = nn.Dropout(self.dropout_prob)

		for ii in range(len(dims)-2):
			self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

			if use_bn:
				self.bns.append(torch.nn.BatchNorm1d(dims[ii+1]))
		self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
		if leaky_relu:
			self.relu = nn.LeakyReLU(negative_slope=0.3)
		else:
			self.relu = torch.nn.ReLU()

		# Initialize linear layers
		if init == "xavier_uniform":
			gain_leaky_relu = nn.init.calculate_gain('leaky_relu', 0.3)
			gain_sigmoid = nn.init.calculate_gain('sigmoid')
			for layer in self.layers:
				nn.init.xavier_uniform_(layer.weight, gain=gain_leaky_relu)
				nn.init.zeros_(layer.bias)
			nn.init.xavier_uniform_(self.layer_output.weight, gain=gain_sigmoid)
			nn.init.zeros_(self.layer_output.bias)
		elif init == "kaiming_uniform":
			for layer in self.layers + [self.layer_output]:
				if leaky_relu:
					nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.3)
				else:
					nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
			nn.init.zeros_(self.layer_output.bias)
		elif init == 'default':
			pass
		else:
			raise NotImplementedError("Unsupported init")

		# Power iteration for spectral norm
		self.sr_u = {}
		self.sr_v = {}
		self.num_power_iter = 4


	def forward(self, x):
		for ii in range(len(self.layers)):
			x = self.layers[ii](x)
			if self.use_bn:
				x = self.bns[ii](x)
			x = self.relu(x)
			if self.dropout_prob > 0:
				x = self.dropout(x)

		return self.layer_output(x)

	def spectral_norm_parallel(self, device):
		"""Code from https://github.com/NVlabs/NVAE/blob/master/model.py
			
		This method computes spectral normalization for all conv layers in parallel. This method should be called
		 after calling the forward method of all the conv layers in each iteration. """

		weights = {}   # a dictionary indexed by the shape of weights
		for ii in range(len(self.layers)):
			weight = self.layers[ii].weight
			weight_mat = weight.view(weight.size(0), -1)

			# Modify by batchnorm?
			if self.use_bn:
				weight_mat = weight_mat * (self.bns[ii].weight.unsqueeze(1) / torch.sqrt(self.bns[ii].running_var.unsqueeze(1)))
			if weight_mat.shape not in weights:
				weights[weight_mat.shape] = []

			weights[weight_mat.shape].append(weight_mat)

		# record the output layer separately, as it's not listed in "layers"
		weight = self.layer_output.weight
		weight_mat = weight.view(weight.size(0), -1)
		if weight_mat.shape not in weights:
			weights[weight_mat.shape] = []
		weights[weight_mat.shape].append(weight_mat)

		loss = 0
		for i in weights:
			weights[i] = torch.stack(weights[i], dim=0)
			with torch.no_grad():
				num_iter = self.num_power_iter
				if i not in self.sr_u:
					num_w, row, col = weights[i].shape
					self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).to(device), dim=1, eps=1e-3)
					self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).to(device), dim=1, eps=1e-3)
					# increase the number of iterations for the first time
					num_iter = 10 * self.num_power_iter

				for j in range(num_iter):
					# Spectral norm of weight equals to `u^T W v`, where `u` and `v`
					# are the first left and right singular vectors.
					# This power iteration produces approximations of `u` and `v`.
					self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
											   dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
					self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
											   dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

			sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
			loss += torch.sum(sigma)
		return loss


#---------------------------------------------------
# Wrapper for MLP and LipMLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
#---------------------------------------------------
# define model
class mlp_wrapper(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, vectorized='true', pos_enc='early',
				 lipschitz=False, one_hot=False, use_bn=False, dropout_prob=0.0,
				 leaky_relu=False, train_x=None,
				 min_temp=10, max_temp=109, init="xavier_uniform", width=128):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)

		If train_x is provided, use this to rescale the input. Specifically, compute mean/std
		for non-categorical variables as train_x[:, self.non_categorical_indices, 0, 0].mean(dim=0).
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.vectorized = vectorized
		self.pos_enc = pos_enc

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

		# Standardize input to (mean 0, std 1) if desired
		if train_x is not None:
			train_features = train_x[:, self.non_categorical_indices, 0, 0]
			self.input_mean = train_features.mean(dim=0, keepdim=True)
			self.input_std = train_features.std(dim=0, keepdim=True)
		else:
			self.input_mean = None
			self.input_std = None

		# Spatial Encoder from PE-GNN
		if pos_enc == "early":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=128,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # TODO True # Enable feedforward network for final spatial embeddings
			)
			self.new_input_size += 128  # Add the spatial embeddings
		elif pos_enc == "late":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=self.num_params,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # TODO True # Enable feedforward network for final spatial embeddings
			)

		# MLP backbone
		if lipschitz:
			self.mlp = lipmlp((self.new_input_size, width, width, self.num_params),
							   use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, leaky relu, dropout, etc. not supported
		else:
			self.mlp = mlp((self.new_input_size, width, width, self.num_params),  # TEMP TODO @joshuafan, 256, 256
							  use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu, init=init)

		# sigmoid parameter
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.min_temp = min_temp
		self.max_temp = max_temp
		self.sigmoid = nn.Sigmoid()


	def forward(self, input_var, wosis_depth, coords, whether_predict):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# For some reason spatial encoder requires coords to have shape [batch, 1, 2] 
		coords = coords.unsqueeze(1).detach()

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			idx = int(idx)
			if self.one_hot:
				emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings. TODO Reconsider this
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)

		# Preprocess numeric (non-categorical) features
		features = predictor[:, self.non_categorical_indices]  # [batch, n_features]
		if self.input_mean is not None and self.input_std is not None:
			features = (features - self.input_mean) / self.input_std

		# Combine numeric features and categorical embeddings
		new_input = torch.concatenate([features, all_embs], dim=1)

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1)  # Remove the channel dimension. [batch, params]
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)

		# check if new_input is nan
		if torch.isnan(new_input).any() or torch.isinf(new_input).any():
			print("new_input was nan", new_input)
			exit(1)

		# Pass through MLP
		self.new_input = new_input
		mlp_output = self.mlp(new_input)

		# check if mlp output is nan
		if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
			print("mlp_output was nan", mlp_output)
			exit(1)

		# Clamp temp_sigmoid to be within a range
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * self.sigmoid(self.temp_sigmoid)  # 10 + 90*self.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			mlp_output += spatial_embeddings

		# Pass parameters through sigmoid to constrain their range
		predicted_para = self.sigmoid(mlp_output / clamped_temp_sigmoid)

		# check if h5 is nan
		if torch.isnan(predicted_para).any() or torch.isinf(predicted_para).any():
			print("predicted_para was nan", predicted_para)
			exit(1)

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_model_prediction(predicted_para, forcing, self.vertical_mixing, self.vectorized)
		else:
			simu_soc = fun_model_simu(predicted_para, forcing, obs_depth, self.vertical_mixing, self.vectorized)
		return simu_soc, predicted_para




#---------------------------------------------------
# Pure NN without process-based model
#---------------------------------------------------
class nn_only(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, pos_enc, output_dim=140,
				 lipschitz=False, one_hot=False, use_bn=False, dropout_prob=0.0,
				 leaky_relu=False, output_mean=None, output_std=None, train_x=None,
				 init="xavier_uniform", width=128):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)

		If output_mean and output_std are provided, uses them to rescale the output.
		If train_x is provided, use this to rescale the input. Specifically, compute mean/std
		for non-categorical variables as train_x[:, self.non_categorical_indices, 0, 0].mean(dim=0).
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.pos_enc = pos_enc
		self.output_dim = output_dim

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

		# Number of parameters (totally fake)
		self.num_params = 22

		# Standardize input to (mean 0, std 1) if desired
		if train_x is not None:
			train_features = train_x[:, self.non_categorical_indices, 0, 0]
			self.input_mean = train_features.mean(dim=0, keepdim=True)
			self.input_std = train_features.std(dim=0, keepdim=True)
		else:
			self.input_mean = None
			self.input_std = None

		# Output transformation
		self.output_mean = output_mean
		self.output_std = output_std

		# Spatial Encoder from PE-GNN
		if pos_enc == "early":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=128,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # TODO True # Enable feedforward network for final spatial embeddings
			)
			self.new_input_size += 128  # Add the spatial embeddings
		elif pos_enc == "late":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=self.num_params,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # TODO True # Enable feedforward network for final spatial embeddings
			)

		# MLP backbone
		if lipschitz:
			self.mlp = lipmlp((self.new_input_size, width, width, self.num_params),
							   use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, leaky relu, dropout, etc. not supported
		else:
			self.mlp = mlp((self.new_input_size, width, width, self.num_params),
							use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu, init=init)
		
		# Final layer: "params" -> SOC pools
		self.final_layer = nn.Linear(self.num_params, self.output_dim)
		if init == "xavier_uniform":
			nn.init.xavier_uniform_(self.final_layer.weight)
			nn.init.zeros_(self.final_layer.bias)
		elif init == "kaiming_uniform":
			nn.init.kaiming_uniform_(self.final_layer.weight)
			nn.init.zeros_(self.final_layer.bias)

		# sigmoid parameter
		self.sigmoid = nn.Sigmoid()


	def forward(self, input_var, wosis_depth, coords, whether_predict,
				return_spatial_embedding=False, **kwargs):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# For some reason spatial encoder requires coords to have shape [batch, 1, 2] 
		coords = coords.unsqueeze(1).detach()

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			idx = int(idx)
			if self.one_hot:
				emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)

		# Preprocess numeric (non-categorical) features
		features = predictor[:, self.non_categorical_indices]  # [batch, n_features]
		if self.input_mean is not None and self.input_std is not None:
			features = (features - self.input_mean) / self.input_std

		# Combine numeric features and categorical embeddings
		new_input = torch.concatenate([features, all_embs], dim=1)

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			# print("Spatial emb", spatial_embeddings.shape, "new input", new_input.shape, "coords", coords.shape)
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)
		elif self.pos_enc == "late":
			raise ValueError("Late pos_enc not supported for nn_only")

		# check if new_input is nan
		if torch.isnan(new_input).any() or torch.isinf(new_input).any():
			print("new_input was nan", new_input)
			exit(1)

		# Pass through MLP to get fake pred_para
		self.new_input = new_input
		pred_para = self.mlp(new_input)
		pred_output = self.final_layer(F.relu(pred_para))
		if self.output_mean is not None and self.output_std is not None:
			pred_output = pred_output * self.output_std + self.output_mean

		# Convert 140 pools to 20 layers
		pred_output = pred_output.reshape((pred_output.shape[0], 20, 7)).mean(dim=2)

		if whether_predict == 1:
			return pred_output, self.sigmoid(pred_para)
		else:
			simu_soc = select_depth(pred_output, forcing, obs_depth)
			return simu_soc, self.sigmoid(pred_para)

