"""
Code from https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py

Re-implements the Lipschitz regularization in the paper "Learning Smooth Neural Functions 
via Lipschitz Regularization" (Liu et al., SIGGRAPH 2022)
"""

import torch
import math

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, init="abs_row_sum"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters(init=init)

    def initialize_parameters(self, init):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        if init == "one":
            self.c.data = torch.tensor([1.0])
        elif init == "abs_row_sum":
            W = self.weight.data
            W_abs_row_sum = torch.abs(W).sum(1)
            self.c.data = W_abs_row_sum.max() # just a rough initialization
        else:
            raise ValueError(f"Invalid init method {init} (expected 'one' or 'abs_row_sum')")

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def get_avg_scaling(self):
        abs_row_sums = torch.abs(self.weight).sum(1)
        return (self.softplus(self.c) / abs_row_sums).mean()

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.clamp(torch.abs(self.weight).sum(1), min=1e-6)  # avoid division by zero
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

class lipmlp(torch.nn.Module):
    def __init__(self, dims, use_bn=False):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.use_bn = use_bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1], init='one'))
            if use_bn:
                self.bns.append(torch.nn.BatchNorm1d(dims[ii+1]))
        self.layer_output = LipschitzLinear(dims[-2], dims[-1], init='abs_row_sum')
        self.relu = torch.nn.ReLU()


    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        cs = []
        scalings = []
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
            cs.append(self.layers[ii].get_lipschitz_constant().item())
            scalings.append(self.layers[ii].get_avg_scaling().item())
        loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        cs.append(self.layer_output.get_lipschitz_constant().item())
        scalings.append(self.layer_output.get_avg_scaling().item())

        if torch.isinf(loss_lipc) or torch.isnan(loss_lipc):
            print("Lipc is inf!")
            exit(1)
        return loss_lipc, cs, scalings

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            if self.use_bn:
                x = self.bns[ii](x)
            x = self.relu(x)
        return self.layer_output(x)



