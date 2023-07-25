import torch
from .adjoint import AdjointTensor, norm

def wrap_RT_adjoint(R, t, joint=True):
    R_j = torch.eye(9, dtype=t.dtype, device=t.device).view(9, 1, 3, 3)
    t_j = torch.eye(3, dtype=t.dtype, device=t.device).view(3, 1, 3)
    if joint:
        R_j = torch.nn.functional.pad(R_j, (0, 0, 0, 0, 0, 0, 0, 3), value=0.)
        t_j = torch.nn.functional.pad(t_j, (0, 0, 0, 0, 9, 0), value=0.)
    R_adj = AdjointTensor(R, R_j)
    t_adj = AdjointTensor(t, t_j)
    return R_adj, t_adj


def normalize_R_if_necessary(R):
    bsize = R.shape[0]
    assert R.shape == (bsize, 3, 3)
    error = (R @ R.transpose(-1, -2) - torch.eye(3, dtype=R.dtype, device=R.device)).view(bsize, -1).abs().max(dim=-1)[0]
    needs_norm_mask = error > 1e-2
    needs_norm_idx, = torch.where(needs_norm_mask)
    if len(needs_norm_idx) == 0:
        return R
    with torch.no_grad():
        needs_norm_R = R[needs_norm_idx]
        u, _, v = torch.svd(needs_norm_R)
        normalized_R = u @ v.transpose(-1, -2)
        R_out = R.clone()
        R_out[needs_norm_idx] = normalized_R
        return R_out

def normalize_R(R):
    bsize = R.shape[0]
    assert R.shape == (bsize, 3, 3)
    u, d, v = torch.svd(R)
    normalized_R = u @ v.transpose(-1, -2)
    return normalized_R

def normalize_T(t):
    return t.div(t.norm(dim=-1, keepdim=True))

def normalize_RT_if_necessary(R, t):
    return normalize_R_if_necessary(R), normalize_T(t)

def RT_tangent_space(R, t):
    dv = R.device
    R, t = wrap_RT_adjoint(R, t, joint=False)
    t_tangent = norm(t, dim=-1, keepdim=True).to_tangent_space()
    xidx = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long, device=dv)
    yidx = torch.tensor([0, 1, 2, 1, 2, 2], dtype=torch.long, device=dv)
    R_tangent = (R.transpose(-1, -2) @ R)[:, xidx, yidx].to_tangent_space()
    return R_tangent, t_tangent


