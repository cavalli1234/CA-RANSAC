import torch
from .fmat import wrap_F_adjoint, normalize_F_if_necessary, F_tangent_space, normalize_F
from .sampson_error import sparse_sampson_error_adjoint_fmat, dense_sampson_error_adjoint_fmat


def project_jacobian_to_tangent_fmat(jac, F_tangent):
    jac = (jac.unsqueeze(-2) @ F_tangent).squeeze(-2)
    return jac

def reduce_sum_indexed(src, batch_idx):
    payload_shape = src.shape[1:]
    buffer = torch.zeros((batch_idx[-1]+1, *payload_shape), device=src.device, dtype=src.dtype)
    buffer = buffer.scatter_reduce(dim=0, index=batch_idx.view(-1, *(1 for _ in payload_shape)).repeat(1, *payload_shape),
                                   src=src, reduce="sum", include_self=False)
    return buffer

def apply_damping(jtj, damp):
    return jtj + torch.diag_embed(damp.unsqueeze(-1) * jtj.diagonal(dim1=-2, dim2=-1))

def update_solution(F0, F_tangent, jtj, mjtr):
    delta_x = torch.cholesky_solve(mjtr.unsqueeze(-1), torch.linalg.cholesky(jtj)).squeeze(-1)

    # Apply to tangent space
    delta_F = (F_tangent @ delta_x.unsqueeze(-1)).view(-1, 3, 3)

    # Apply delta
    F_prime = F0 + delta_F
    F_prime = normalize_F_if_necessary(F_prime)
    return F_prime

def check_improvement(F0, F_prime, hpts1, hpts2, batch_idx, weights, damp, total_error0, scale, loss_shape, loss_type):
    # Check improvement
    err_prime = sparse_sampson_error_adjoint_fmat(F_prime, hpts1, hpts2, scale.square(), batch_idx, compute_jacobian=False, loss_shape=loss_shape, loss_type=loss_type)
    err_prime = err_prime * weights
    total_error_prime = reduce_sum_indexed(err_prime.square(), batch_idx)

    error_delta = total_error0 - total_error_prime
    improved = error_delta > 0
    # improved, = torch.where(error_delta > 0)
    F_out = F0.clone()
    total_error_out = total_error0.clone()
    damp_out = (damp.clone() * 2.0).clamp(max=1e+3)
    F_out[improved] = F_prime[improved]
    total_error_out[improved] = total_error_prime[improved]
    damp_out[improved] = (damp[improved] * 0.5).clamp(min=1e-9)

    return F_out, damp_out, total_error_out, improved

def compute_error_and_jtj_fmat(F0, hpts1, hpts2, weights, batch_idx, scale, loss_shape, loss_type):
    F_tangent = F_tangent_space(F0)

    jac, err = sparse_sampson_error_adjoint_fmat(F0, hpts1, hpts2,
                                                 scale.square(), batch_idx,
                                                 compute_jacobian=True,
                                                 loss_shape=loss_shape,
                                                 loss_type=loss_type)
    jac = jac * weights.unsqueeze(-1)
    err = err * weights
    total_error = reduce_sum_indexed(err.square(), batch_idx)
    jac = project_jacobian_to_tangent_fmat(jac, F_tangent[batch_idx])

    jtj = reduce_sum_indexed(jac.unsqueeze(-1) * jac.unsqueeze(-2), batch_idx)
    return err, jtj

def compute_error_and_jac_dense_fmat(F0, hpts1, hpts2, scale, loss_shape, loss_type):
    F_tangent = F_tangent_space(F0)

    jac, err = dense_sampson_error_adjoint_fmat(F0, hpts1, hpts2,
                                                scale.square(),
                                                compute_jacobian=True,
                                                loss_shape=loss_shape,
                                                loss_type=loss_type)
    jac = project_jacobian_to_tangent_fmat(jac, F_tangent.unsqueeze(1))

    return err, jac

def sparse_lm_iteration_fresh(F0, hpts1, hpts2, weights, batch_idx, damp, scale, loss_shape, loss_type):
    F_tangent = F_tangent_space(F0)

    jac, err = sparse_sampson_error_adjoint_fmat(F0, hpts1, hpts2,
                                                 scale.square(), batch_idx,
                                                 compute_jacobian=True, loss_shape=loss_shape, loss_type=loss_type)
    # print("Weights: ", [w.item() for w in weights])
    # print("Jacs: ", [j for j in jac])
    jac = jac * weights.unsqueeze(-1)
    err = err * weights
    total_error = reduce_sum_indexed(err.square(), batch_idx)
    jac = project_jacobian_to_tangent_fmat(jac, F_tangent[batch_idx])

    jtj = reduce_sum_indexed(jac.unsqueeze(-1) * jac.unsqueeze(-2), batch_idx)
    mjtr = reduce_sum_indexed(-jac * err.unsqueeze(-1), batch_idx)
    grad_norm = mjtr.norm(dim=-1)

    jtj = apply_damping(jtj, damp)
    F_prime = update_solution(F0, F_tangent, jtj, mjtr)
    F_out, damp_out, total_error_out, improved = check_improvement(F0, F_prime,
                                                                   hpts1, hpts2, batch_idx,
                                                                   weights, damp, total_error,
                                                                   scale, loss_shape, loss_type)

    return F_out, damp_out, jtj, mjtr, total_error_out, improved, grad_norm

def sparse_lm_iteration_mixed(F0, hpts1, hpts2, weights, batch_idx, damp, scale, jtj_prev, mjtr_prev, total_error_prev, batch_mask, loss_shape, loss_type):
    F_tangent = F_tangent_space(F0)

    batch_mask_idx, = torch.where(batch_mask)
    data_mask = (batch_idx.unsqueeze(-1) == batch_mask_idx).any(dim=-1)
    offset_batch_idx = batch_mask.long().cumsum(dim=0) - 1
    local_batch_idx = offset_batch_idx[batch_idx][data_mask]

    weights_ = weights[data_mask]
    jac, err = sparse_sampson_error_adjoint_fmat(F0[batch_mask],
                                                 hpts1[data_mask], hpts2[data_mask],
                                                 scale.square(), local_batch_idx,
                                                 compute_jacobian=True, loss_shape=loss_shape, loss_type=loss_type)
    jac = jac * weights_.unsqueeze(-1)
    err = err * weights_
    total_error_new = reduce_sum_indexed(err.square(), local_batch_idx)
    jac = project_jacobian_to_tangent_fmat(jac, F_tangent[batch_mask][local_batch_idx])

    jtj_new = reduce_sum_indexed(jac.unsqueeze(-1) * jac.unsqueeze(-2), local_batch_idx)
    mjtr_new = reduce_sum_indexed(-jac * err.unsqueeze(-1), local_batch_idx)

    jtj = jtj_prev.clone()
    jtj[batch_mask] = jtj_new
    total_error = total_error_prev.clone()
    total_error[batch_mask] = total_error_new
    mjtr = mjtr_prev.clone()
    mjtr[batch_mask] = mjtr_new
    grad_norm = mjtr.norm(dim=-1)

    jtj = apply_damping(jtj, damp)
    F_prime = update_solution(F0, F_tangent, jtj, mjtr)
    F_out, damp_out, total_error_out, improved = check_improvement(F0, F_prime,
                                                                   hpts1, hpts2, batch_idx,
                                                                   weights, damp, total_error,
                                                                   scale, loss_shape, loss_type)

    return F_out, damp_out, jtj, mjtr, total_error_out, improved, grad_norm


def sparse_lm_iteration_cache_only(F0, hpts1, hpts2, weights, batch_idx, damp, jtj, mjtr, total_error, scale, loss_shape, loss_type):
    F_tangent = F_tangent_space(F0)

    grad_norm = mjtr.norm(dim=-1)

    jtj = apply_damping(jtj, damp)
    F_prime = update_solution(F0, F_tangent, jtj, mjtr)
    F_out, damp_out, total_error_out, improved = check_improvement(F0, F_prime,
                                                                   hpts1, hpts2, batch_idx,
                                                                   weights, damp, total_error,
                                                                   scale, loss_shape, loss_type)
    return F_out, damp_out, jtj, mjtr, total_error_out, improved, grad_norm

def sparse_lm_loop_fmat(F0, hpts1, hpts2, weights, batch_idx, loss_scale=1e-3, loss_shape=None, loss_type='CAUCHY', num_iterations=20, gtol=1e-6):
    bsize = F0.shape[0]
    dt = F0.dtype
    dv = F0.device
    F = normalize_F_if_necessary(F0)
    damps = 1e-4 * torch.ones((bsize,), dtype=dt, device=dv)
    jtj = torch.zeros((bsize, 7, 7), dtype=dt, device=dv)
    mjtr = torch.zeros_like(jtj[..., 0])
    te = torch.zeros_like(mjtr[:, 0])
    improved = torch.ones_like(te).bool()
    with torch.set_grad_enabled(False):
        for i in range(num_iterations):
            if torch.all(improved):
                F, damps, jtj, mjtr, te, improved, grad_norm = sparse_lm_iteration_fresh(F, hpts1, hpts2, weights, batch_idx, damps, loss_scale, loss_shape, loss_type)
            elif torch.all(~improved):
                F, damps, jtj, mjtr, te, improved, grad_norm = sparse_lm_iteration_cache_only(F, hpts1, hpts2, weights, batch_idx, damps, jtj, mjtr, te, loss_scale, loss_shape, loss_type)
            else:
                F, damps, jtj, mjtr, te, improved, grad_norm = sparse_lm_iteration_mixed(F, hpts1, hpts2, weights, batch_idx, damps, loss_scale, jtj, mjtr, te, improved, loss_shape, loss_type)

            # print("Sparse lm iter error: ", te)
            # print("Sparse LM iter damps: ", damps)
            # print("Sparse LM iter improved: ", improved)
            if grad_norm.max().item() < gtol:
                break
    if any([hpts1.requires_grad, hpts2.requires_grad, weights.requires_grad, loss_scale.requires_grad]):
        if torch.all(improved):
            F, damps, jtj, mjtr, te, improved, grad_norm = sparse_lm_iteration_fresh(F, hpts1, hpts2, weights, batch_idx, damps, loss_scale, loss_shape, loss_type)
        elif torch.all(~improved):
            F, damps, jtj, mjtr, te, improved, grad_norm = sparse_lm_iteration_cache_only(F, hpts1, hpts2, weights, batch_idx, damps, jtj, mjtr, te, loss_scale, loss_shape, loss_type)
        else:
            F, damps, jtj, mjtr, te, improved, grad_norm = sparse_lm_iteration_mixed(F, hpts1, hpts2, weights, batch_idx, damps, loss_scale, jtj, mjtr, te, improved, loss_shape, loss_type)
    return F

