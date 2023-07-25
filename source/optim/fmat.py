import torch
from utils import batch_3x3_det
from .adjoint import AdjointTensor, det3x3

def wrap_F_adjoint(F):
    F_j = torch.eye(9, dtype=F.dtype, device=F.device).view(9, 1, 3, 3).expand(9, F.shape[0], 3, 3)
    F_adj = AdjointTensor(F, F_j)
    return F_adj

def F_tangent_space(F):
    dv = F.device
    F = wrap_F_adjoint(F)
    F_det = det3x3(F)
    F_sum = F.sum(dim=-1).sum(dim=-1)
    F_constr = AdjointTensor(torch.stack([F_det.val, F_sum.val], dim=-1), torch.stack([F_det.jac, F_sum.jac], dim=-1))
    F_tangent = F_constr.to_tangent_space()
    return F_tangent

def normalize_F_if_necessary(F):
    bsize = F.shape[0]
    assert F.shape == (bsize, 3, 3)
    F = F.div(F.sum(dim=(-1, -2), keepdim=True))
    error = batch_3x3_det(F).flatten().abs()
    needs_norm_mask = error > 1e-4
    needs_norm_idx, = torch.where(needs_norm_mask)
    if len(needs_norm_idx) == 0:
        return F
    with torch.no_grad():
        needs_norm_F = F[needs_norm_idx]
        u, d, v = torch.svd(needs_norm_F)
        d[:, -1] = 0.
        normalized_F = u @ torch.diag_embed(d) @ v.transpose(-1, -2)
    F_out = F.clone()
    F_out[needs_norm_idx] = normalized_F
    return F_out

def normalize_F(F):
    F = F.div(F.sum(dim=(-1, -2), keepdim=True))
    u, d, v = torch.svd(F)
    d[-1] = 0.
    normalized_F = u @ torch.diag(d) @ v.transpose(-1, -2)
    return normalized_F


