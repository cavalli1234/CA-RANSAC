from collections import namedtuple
from functools import wraps
from utils import cross_mat as torch_cross_mat
from utils import normal_space
import torch
import time

_AdjointTensor = namedtuple("AdjointTensor", ["val", "jac"])

def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a != b and b != 1 and a != 1:
            return False
    return True

class AdjointTensor(_AdjointTensor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        assert self.val.ndim + 1 == self.jac.ndim, f"Error. Found incompatible value and jacobian shapes (respectively {self.val.shape} and {self.jac.shape})."
        assert is_broadcastable(self.val.shape, self.jac.shape[1:]), f"Error. Found incompatible value and jacobian shapes (respectively {self.val.shape} and {self.jac.shape})."


    def __add__(self, other):
        if type(other) == AdjointTensor:
            return add(self, other)
        return add_const(self, other)

    def __sub__(self, other):
        if type(other) == AdjointTensor:
            return sub(self, other)
        return sub_const(self, other)

    def __mul__(self, other):
        if type(other) == AdjointTensor:
            return multiply(self, other)
        return multiply_const(self, other)

    def __matmul__(self, other):
        if type(other) == AdjointTensor:
            return matmul(self, other)
        return matmul_const(self, other)

    def __truediv__(self, other):
        if type(other) == AdjointTensor:
            return divide(self, other)
        return divide_const(self, other)

    def __pow__(self, other):
        if type(other) == AdjointTensor:
            raise NotImplementedError
        return const_power(self, other)

    def __getitem__(self, indexing):
        if type(indexing) is tuple:
            jac_indexing = (slice(None, None, None), *indexing)
        else:
            jac_indexing = (slice(None, None, None), indexing)
        return AdjointTensor(self.val[indexing], self.jac[jac_indexing])

    def square(self):
        return const_power(self, 2)

    def sqrt(self):
        return const_power(self, 0.5)

    def reciprocal(self):
        return reciprocal(self)

    def log(self):
        return log(self)

    def log1p(self):
        return log1p(self)

    def abs(self):
        return abs(self)

    def exp(self):
        return exp(self)

    def transpose(self, dim0, dim1):
        return transpose(self, dim0, dim1)

    def unsqueeze(self, dim):
        dim = dim - self.val.ndim - 1 if dim >= 0 else dim
        return AdjointTensor(self.val.unsqueeze(dim), self.jac.unsqueeze(dim))

    def squeeze(self, dim):
        dim = dim - self.val.ndim - 1 if dim >= 0 else dim
        return AdjointTensor(self.val.squeeze(dim), self.jac.squeeze(dim))

    def zeros_like(self):
        return AdjointTensor(torch.zeros_like(self.val), torch.zeros_like(self.jac))

    def sum(self, dim, keepdim=False):
        return reduce_sum(self, dim=dim, keepdim=keepdim)

    def clamp_max(self, value):
        return AdjointTensor(self.val.clamp(max=value), self.jac * (self.val < value).unsqueeze(0).float())

    def clamp_min(self, value):
        return AdjointTensor(self.val.clamp(min=value), self.jac * (self.val > value).unsqueeze(0).float())

    def scatter_reduce_sum(self, dim, index, source, include_self):
        dim = dim - self.val.ndim - 1 if dim >= 0 else dim
        return AdjointTensor(self.val.scatter_reduce(dim, index, source, "sum", include_self=include_self),
                             self.jac.scatter_reduce(dim, index, source, "sum", include_self=include_self))

    def to_tangent_space(self):
        j = self.jac
        n_dims, bsize = j.shape[:2]
        j = j.view(n_dims, bsize, -1)
        n_constr = j.shape[-1]
        assert n_constr < n_dims
        return normal_space(j.transpose(0, 1))

    @property
    def shape(self):
        return self.val.shape, self.jac.shape


def to_adjoint(x):
    if type(x) == AdjointTensor:
        return x
    if type(x) == torch.Tensor:
        return AdjointTensor(x, torch.zeros_like(x).unsqueeze(0))
    raise NotImplementedError

def cast_inputs(func):
    @wraps
    def wrapper(*args, **kwargs):
        args = (to_adjoint(a) for a in args)
        kwargs = {k: to_adjoint(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return func

def verify(adjoint_func, pytorch_func, example_input):
    assert example_input.ndim == 1
    ex_t, ex_j = example_input, torch.eye(example_input.shape[0])
    t1 = time.time()
    adj_y = adjoint_func(AdjointTensor(ex_t, ex_j))
    t2 = time.time()
    y = pytorch_func(example_input)
    autograd_j = torch.autograd.functional.jacobian(pytorch_func, example_input)
    t3 = time.time()
    assert torch.max(adj_y.val - y) < 1e-6
    assert torch.max(adj_y.jac - autograd_j.T) < 1e-6
    print(t2-t1, t3-t2)

# @cast_inputs
def cross_mat(x):
    return AdjointTensor(torch_cross_mat(x.val), torch_cross_mat(x.jac))

# @cast_inputs
def RT_to_E(R, t):
    return matmul(cross_mat(normalize(t, dim=-1)), R)

# @cast_inputs
def matmul(A, B):
    return AdjointTensor(A.val @ B.val, A.jac @ B.val + A.val @ B.jac)

# @cast_inputs
def matmul_const(A, K):
    return AdjointTensor(A.val @ K, A.jac @ K)

def normalize(v, dim):
    # v = to_adjoint(v)
    return divide(v, norm(v, dim=dim, keepdim=True))

def pad(x, pad, value):
    # x = to_adjoint(x)
    return AdjointTensor(torch.nn.functional.pad(x.val, pad, value=value),
                         torch.nn.functional.pad(x.jac, pad, value=0.))

def transpose(x, dim0, dim1):
    # x = to_adjoint(x)
    dim0, dim1 = (d - x.ndim - 1 if d >= 0 else d for d in [dim0, dim1])
    return AdjointTensor(x.val.transpose(dim0, dim1), x.jac.transpose(dim0, dim1))

# @cast_inputs
def divide(a, b):
    return multiply(a, reciprocal(b))

def divide_const(a, b):
    return multiply_const(a, 1/b)

# @cast_inputs
def multiply(a, b):
    return AdjointTensor(a.val * b.val, a.jac * b.val + a.val * b.jac)

def multiply_const(a, k):
    # a = to_adjoint(a)
    return AdjointTensor(a.val * k, a.jac * k)

# @cast_inputs
def add(a, b):
    return AdjointTensor(a.val + b.val, a.jac + b.jac)

def add_const(a, k):
    # a = to_adjoint(a)
    return AdjointTensor(a.val + k, a.jac)

# @cast_inputs
def sub(a, b):
    return AdjointTensor(a.val - b.val, a.jac - b.jac)

def sub_const(a, k):
    # a = to_adjoint(a)
    return AdjointTensor(a.val - k, a.jac)

def norm(x, dim=-1, keepdim=False):
    # x = to_adjoint(x)
    return const_power(reduce_sum(const_power(x, 2), dim=dim, keepdim=keepdim), 0.5)

def det3x3(x):
    v00 = x.val[..., 0, 0]
    v01 = x.val[..., 0, 1]
    v02 = x.val[..., 0, 2]
    v10 = x.val[..., 1, 0]
    v11 = x.val[..., 1, 1]
    v12 = x.val[..., 1, 2]
    v20 = x.val[..., 2, 0]
    v21 = x.val[..., 2, 1]
    v22 = x.val[..., 2, 2]
    j00 = x.jac[..., 0, 0]
    j01 = x.jac[..., 0, 1]
    j02 = x.jac[..., 0, 2]
    j10 = x.jac[..., 1, 0]
    j11 = x.jac[..., 1, 1]
    j12 = x.jac[..., 1, 2]
    j20 = x.jac[..., 2, 0]
    j21 = x.jac[..., 2, 1]
    j22 = x.jac[..., 2, 2]

    m00 = v11 * v22 - v21 * v12
    m01 = v10 * v22 - v12 * v20
    m02 = v10 * v21 - v11 * v20
    m10 = v01 * v22 - v21 * v02
    m11 = v00 * v22 - v02 * v20
    m12 = v00 * v21 - v01 * v20
    m20 = v01 * v12 - v11 * v02
    m21 = v00 * v12 - v02 * v10
    m22 = v00 * v11 - v01 * v10

    det = v00 * m00 - \
          v01 * m01 + \
          v02 * m02

    det_jac = j00 * m00 - j01 * m01 + j02 * m02 - \
              j10 * m10 + j11 * m11 - j12 * m12 + \
              j20 * m20 - j21 * m21 + j22 * m22

    return AdjointTensor(det, det_jac)

# @cast_inputs
def log(x):
    return AdjointTensor(x.val.log(), x.jac / x.val)

# @cast_inputs
def log1p(x):
    return AdjointTensor(torch.log1p(x.val), x.jac / (x.val + 1.))

# @cast_inputs
def abs(x):
    neg_mask = (x.val < 0).to(x.val.dtype) * (-2) + 1
    return AdjointTensor(x.val.abs(), x.jac * neg_mask)

# @cast_inputs
def exp(x):
    expx = x.val.exp()
    return AdjointTensor(expx, expx * x.jac)

def reduce_sum(x, dim, keepdim=False):
    # x = to_adjoint(x)
    if dim >= 0:
        dim = dim - x.ndim - 1
    return AdjointTensor(x.val.sum(dim=dim, keepdim=keepdim), x.jac.sum(dim=dim, keepdim=keepdim))

# @cast_inputs
def reciprocal(x):
    return const_power(x, -1)

def const_power(x, n):
    if n == 1:
        return x
    elif n == 0:
        return AdjointTensor(torch.zeros_like(x.val), torch.zeros_like(x.jac))
    # x = to_adjoint(x)
    return AdjointTensor(x.val ** n, n * (x.val ** (n-1)) * x.jac)

if __name__ == '__main__':
    def f_torch(x):
        return x[:40].div(torch.linalg.norm(x))

    def f_adj(x):
        return x[:40] / norm(x, dim=-1, keepdim=True)

    for _ in range(10):
        verify(f_adj, f_torch, torch.rand(500))
