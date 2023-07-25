import numpy as np
import torch
from torch import nn
from torch.nn.functional import leaky_relu



# ------------ BASE CLASSES ----------------
class ClonableModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init_args = args
        self.init_kwargs = kwargs

    def clone(self, with_parameters=True):
        copy = type(self)(*self.init_args, **self.init_kwargs)
        if with_parameters:
            copy.load_state_dict(self.state_dict())
        return copy

class MLP(nn.Module):
    def __init__(self, dims, hidden_act=leaky_relu, final_act=None, bias=True, ln=False, residual=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(len(dims) - 1)])
        if ln:
            self.lnorms = nn.ModuleList(
                [torch.nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)])

        self.hidden_act = hidden_act
        self.final_act = final_act
        self.ln = ln
        self.residual = residual

    def forward(self, x):
        original_shape = x.shape
        channels = original_shape[-1]
        x = x.view(-1, channels)
        for i, lin in enumerate(self.layers):
            y = lin(x)
            if self.ln:
                y = self.lnorms[i](y)
            if i+1 < len(self.layers):
                y = self.hidden_act(y)
            if self.residual and x.shape == y.shape:
                x = y + x
            else:
                x = y
        if self.final_act is not None:
            x = self.final_act(x)
        return x.view(*original_shape[:-1], -1)


class CoordinateEmbedding(nn.Module):
    def __init__(self, num_frequencies=4, rng=1.):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.rng = rng
        self.dim_multiplier = self.num_frequencies * 2

    def forward(self, x):
        xpow = [x * (np.power(2, i) * np.pi) / self.rng for i in range(self.num_frequencies)]
        return torch.cat([torch.sin(xp) for xp in xpow] + [torch.cos(xp) for xp in xpow], dim=-1)

# ------------ UTILITY MODULES ----------------

class LearnedMergerModule(ClonableModule):
    def __init__(self, m1, m2, init_logw=2.0):
        super().__init__(m1, m2, init_logw)
        self.m1 = m1
        self.m2 = m2
        self.base_log_weight = nn.Parameter(torch.ones((1,), dtype=torch.float32) * init_logw, requires_grad=True)

    def forward(self, *args, **kwargs):
        o1 = self.m1(*args, **kwargs)
        o2 = self.m2(*args, **kwargs)
        w = self.base_log_weight.sigmoid()
        return w * o1 + (1-w) * o2

class ConstantModule(ClonableModule):
    def __init__(self, k):
        super().__init__(k)
        self.k = k

    def forward(self, points, *args, **kwargs):
        return torch.tensor(self.k, device=points.device, dtype=points.dtype)

# ------------ FEATURE INITIALIZERS ----------------

class TrivialInitializer(ClonableModule):
    def __init__(self, dim=1):
        super().__init__(dim)
        self.dim = dim

    def forward(self, points, *args, **kwargs):
        n_pts = points.shape[0]
        return torch.ones(size=(n_pts, self.dim), dtype=points.dtype, device=points.device) / np.sqrt(self.dim)


class SideMLPInitializer(ClonableModule):
    def __init__(self, dims, embedder=None):
        super().__init__(dims, embedder)
        dims = [d for d in dims]
        self.embedder = embedder
        if embedder is not None:
            dims[0] = dims[0] * embedder.dim_multiplier
        self.dims = dims
        self.mlp = MLP(dims)

    def forward(self, points, sides, *args, **kwargs):
        if self.embedder is not None:
            sides = self.embedder(sides)
        feats = self.mlp(sides)
        return feats

# ------------ INLIER PROBABILITY DECODERS ----------------

class TrivialProbabilityModel(ClonableModule):

    def forward(self, features, *args, **kwargs):
        return features[:, 0].clamp(0., 1.)

class TrivialLogProbabilityModel(ClonableModule):

    def forward(self, features, *args, **kwargs):
        return features[:, 0].sigmoid()

class NormProbabilityModel(ClonableModule):

    def forward(self, features, *args, **kwargs):
        return features.norm(dim=-1).tanh()


class MLPProbabilityModel(ClonableModule):
    def __init__(self, f_dim, n_layers=1):
        super().__init__(f_dim, n_layers)
        self.mlp = MLP([f_dim // (2 ** i) for i in range(n_layers)] + [1], final_act=torch.sigmoid)

    def forward(self, features, points, *args, **kwargs):
        prob = self.mlp(features).squeeze(-1)
        return prob

# ------------ FEATURE UPDATE MODULES ----------------

class TrivialFeatureUpdateModule(ClonableModule):

    def forward(self, features, score_matrix, *args, **kwargs):
        return score_matrix.T @ (score_matrix @ features)


class MLPFeatureUpdateModule(ClonableModule):
    def __init__(self, f_dim, n_layers=3, residual=True):
        super().__init__(f_dim, n_layers, residual)
        self.mlp1 = MLP([f_dim] * (n_layers+1), hidden_act=torch.tanh)
        self.mlp2 = MLP([f_dim] * (n_layers+1), hidden_act=torch.tanh)
        self.residual = residual

    def forward(self, features, score_matrix, sides, *args, **kwargs):
        f2 = self.mlp1(features)
        aggr_f2 = score_matrix.T @ (score_matrix @ f2)
        f3 = self.mlp2(aggr_f2)
        if self.residual:
            out = features + f3
        else:
            out = f3
        return out

class FusionMLPFeatureUpdateModule(ClonableModule):
    def __init__(self, f_dim, n_layers=3):
        super().__init__(f_dim, n_layers)
        self.mlp1 = MLP([f_dim] * (n_layers+1), hidden_act=torch.tanh)
        self.mlp2 = MLP([f_dim] * (n_layers+1), hidden_act=torch.tanh)
        self.mlp3 = MLP([f_dim * 2] * n_layers + [f_dim])

    def forward(self, features, score_matrix, sides, *args, **kwargs):
        f2 = self.mlp1(features)
        aggr_f2 = score_matrix.T @ (score_matrix @ f2)
        f3 = self.mlp2(aggr_f2)
        mlp3_in = torch.cat([f3, features], dim=-1)
        out = self.mlp3(mlp3_in)
        return out

