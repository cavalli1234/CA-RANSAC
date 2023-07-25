import torch
from utils import cross_mat
from .adjoint import AdjointTensor, RT_to_E
from .rt import wrap_RT_adjoint
from .fmat import wrap_F_adjoint

def weighted_sampson_error(R, t, hom_pts1, hom_pts2, weights):
    models = cross_mat(t.div(t.norm(dim=-1, keepdim=True))) @ R

    # calculate the sampson distance and msac scores
    M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)  # n_models, n_pts
    squared_distances = x1_M_x2_.square().div(JJ_T_)
    cauchy_loss_residuals = torch.log1p(squared_distances.div(1e-6)) + 1e-6
    return cauchy_loss_residuals.sqrt() * weights

def sampson_error(R, t, hom_pts1, hom_pts2, K1, K2):
    models = torch.inverse(K2).transpose(-1, -2) @ cross_mat(t.div(t.norm(dim=-1, keepdim=True))) @ R @ torch.inverse(K1)

    # calculate the sampson distance and msac scores
    M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)  # n_models, n_pts
    dist = x1_M_x2_.div(JJ_T_.sqrt())
    return dist

def weighted_sampson_error_adjoint(R, t, hom_pts1, hom_pts2, weights, scale):
    R, t = wrap_RT_adjoint(R, t)
    models = RT_to_E(R, t)

    # calculate the sampson distance and msac scores
    M_x1_ = models @ hom_pts1.transpose(-1, -2)  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2) @ hom_pts2.transpose(-1, -2)  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = (M_x2_ * hom_pts1.T.unsqueeze(0)).sum(-2)  # n_models, n_pts
    squared_distances = x1_M_x2_.square() / JJ_T_
    # squared_distances_log = x1_M_x2_.abs().log() * 2 - JJ_T_.log()
    cauchy_loss_residuals = (squared_distances / scale.square()).log1p()
    # cauchy_loss_residuals = ((squared_distances_log - scale.log() * 2).exp() + 1.).log()
    weighted_residuals = cauchy_loss_residuals.sqrt() * weights
    error, jac = weighted_residuals.val, weighted_residuals.jac
    jac = jac.unsqueeze(-1).transpose(0, -1).squeeze(0)

    return jac, error

def dense_sampson_error_adjoint_fmat(F, hom_pts1, hom_pts2, sqrscale, compute_jacobian=True, loss_shape=None, loss_type='CAUCHY'):
    if compute_jacobian:
        models = wrap_F_adjoint(F)
    else:
        models = F

    # calculate the sampson distance and msac scores
    M_x1_ = models @ hom_pts1.transpose(-1, -2)  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2) @ hom_pts2.transpose(-1, -2)  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = (M_x2_ * hom_pts1.T.unsqueeze(0)).sum(-2)  # n_models, n_pts
    squared_distances = x1_M_x2_.square() / JJ_T_
    if loss_type == 'TRUNCATED':
        loss_residuals = (squared_distances / sqrscale).clamp_max(1.).sqrt()
    elif loss_type == 'BARRON':
        abs_shape_minus_two = (loss_shape - 2).abs()
        loss_residuals = (((squared_distances / sqrscale) / abs_shape_minus_two + 1.0) ** (loss_shape/2) - 1.0) * abs_shape_minus_two.div(loss_shape)
    elif loss_type == 'CAUCHY':
        loss_residuals = (squared_distances / sqrscale).log1p().sqrt()
    else:
        raise NotImplementedError(f"Loss type {loss_type} unknown.")

    if compute_jacobian:
        error, jac = loss_residuals.val, loss_residuals.jac
        jac = jac.unsqueeze(-1).transpose(0, -1).squeeze(0)

        return jac, error
    return loss_residuals

def dense_sampson_error_adjoint(R, t, hom_pts1, hom_pts2, sqrscale, compute_jacobian=True, loss_shape=None, loss_type='CAUCHY'):
    if compute_jacobian:
        R, t = wrap_RT_adjoint(R, t)
        models = RT_to_E(R, t)  # (bsize, 3, 3)
    else:
        models = cross_mat(t) @ R

    # calculate the sampson distance and msac scores
    M_x1_ = models @ hom_pts1.transpose(-1, -2)  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2) @ hom_pts2.transpose(-1, -2)  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = (M_x2_ * hom_pts1.T.unsqueeze(0)).sum(-2)  # n_models, n_pts
    squared_distances = x1_M_x2_.square() / JJ_T_
    if loss_type == 'TRUNCATED':
        loss_residuals = (squared_distances / sqrscale).clamp_max(1.).sqrt()
    elif loss_type == 'BARRON':
        abs_shape_minus_two = (loss_shape - 2).abs()
        loss_residuals = (((squared_distances / sqrscale) / abs_shape_minus_two + 1.0) ** (loss_shape/2) - 1.0) * abs_shape_minus_two.div(loss_shape)
    elif loss_type == 'CAUCHY':
        loss_residuals = (squared_distances / sqrscale).log1p().sqrt()
    else:
        raise NotImplementedError(f"Loss type {loss_type} unknown.")

    if compute_jacobian:
        error, jac = loss_residuals.val, loss_residuals.jac
        jac = jac.unsqueeze(-1).transpose(0, -1).squeeze(0)

        return jac, error
    return loss_residuals

def sparse_sampson_error_adjoint(R, t, hom_pts1, hom_pts2, sqrscale, batch_idx, compute_jacobian=True, loss_shape=None, loss_type='CAUCHY'):
    if compute_jacobian:
        R, t = wrap_RT_adjoint(R, t)
        models = RT_to_E(R, t)  # (bsize, 3, 3)
    else:
        models = cross_mat(t) @ R
    models_ = models[batch_idx]  # (n_residuals, 3, 3)

    # calculate the sampson distance and msac scores
    M_x1_ = (models_ @ hom_pts1.unsqueeze(-1)).squeeze(-1)  # n_residuals, 3
    M_x2_ = (models_.transpose(-1, -2) @ hom_pts2.unsqueeze(-1)).squeeze(-1)  # n_residuals, 3
    JJ_T_ = M_x1_[:, 0].square() + M_x1_[:, 1].square() + M_x2_[:, 0].square() + M_x2_[:, 1].square()  # n_residuals,
    x1_M_x2_ = (M_x2_ * hom_pts1).sum(-1)  # n_residuals,
    squared_distances = x1_M_x2_.square() / JJ_T_
    if loss_type == 'TRUNCATED':
        loss_residuals = (squared_distances / sqrscale).clamp_max(1.).sqrt()
    elif loss_type == 'BARRON':
        abs_shape_minus_two = (loss_shape - 2).abs()
        loss_residuals = (((squared_distances / sqrscale) / abs_shape_minus_two + 1.0) ** (loss_shape/2) - 1.0) * abs_shape_minus_two.div(loss_shape)
    elif loss_type == 'CAUCHY':
        loss_residuals = (squared_distances / sqrscale).log1p().sqrt()
    else:
        raise NotImplementedError(f"Loss type {loss_type} unknown.")

    if compute_jacobian:
        error, jac = loss_residuals.val, loss_residuals.jac
        jac = jac.unsqueeze(-1).transpose(0, -1).squeeze(0)

        return jac, error
    return loss_residuals

def sparse_sampson_error_adjoint_fmat(F, hom_pts1, hom_pts2, sqrscale, batch_idx, compute_jacobian=True, loss_shape=None, loss_type='CAUCHY'):
    if compute_jacobian:
        models = wrap_F_adjoint(F)
    else:
        models = F
    models_ = models[batch_idx]  # (n_residuals, 3, 3)

    # calculate the sampson distance and msac scores
    M_x1_ = (models_ @ hom_pts1.unsqueeze(-1)).squeeze(-1)  # n_residuals, 3
    M_x2_ = (models_.transpose(-1, -2) @ hom_pts2.unsqueeze(-1)).squeeze(-1)  # n_residuals, 3
    JJ_T_ = M_x1_[:, 0].square() + M_x1_[:, 1].square() + M_x2_[:, 0].square() + M_x2_[:, 1].square()  # n_residuals,
    x1_M_x2_ = (M_x2_ * hom_pts1).sum(-1)  # n_residuals,
    squared_distances = x1_M_x2_.square() / JJ_T_
    if loss_type == 'TRUNCATED':
        loss_residuals = (squared_distances / sqrscale).clamp_max(1.).sqrt()
    elif loss_type == 'BARRON':
        abs_shape_minus_two = (loss_shape - 2).abs()
        loss_residuals = (((squared_distances / sqrscale) / abs_shape_minus_two + 1.0) ** (loss_shape/2) - 1.0) * abs_shape_minus_two.div(loss_shape)
    elif loss_type == 'CAUCHY':
        loss_residuals = (squared_distances / sqrscale).log1p().sqrt()
    else:
        raise NotImplementedError(f"Loss type {loss_type} unknown.")

    if compute_jacobian:
        error, jac = loss_residuals.val, loss_residuals.jac
        jac = jac.unsqueeze(-1).transpose(0, -1).squeeze(0)

        return jac, error
    return loss_residuals
