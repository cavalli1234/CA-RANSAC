from .essential_matrix_estimator import decompose_and_triangulate, EssentialMatrixEstimator, triangulate_points_and_models
from .fundamental_matrix_estimator import cheirality_check as check_cheirality_fmat
from poselib import refine_fundamental, refine_relative_pose, CameraPose
from optim.lm import sparse_lm_loop
from optim.fmat_lm import sparse_lm_loop_fmat
from utils import RT_to_E
import torch
import numpy as np



def compute_squared_distances(models, points):
    pts1 = points[:, 0:2]
    pts2 = points[:, 2:4]
    dev = points.device
    assert dev == models.device

    num_pts = pts1.shape[0]
    #truncated_threshold = 3 / 2 * threshold  # wider threshold

    # get homogenous coordinates
    hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=dev)), dim=-1)
    hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=dev)), dim=-1)

    models = RT_to_E(models)

    # calculate the sampson distance and msac scores
    M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)  # n_models, n_pts
    squared_distances = x1_M_x2_.square().div(JJ_T_)
    return squared_distances

def compute_squared_distances_fmat(models, points):
    pts1 = points[:, 0:2]
    pts2 = points[:, 2:4]
    dev = points.device
    assert dev == models.device

    num_pts = pts1.shape[0]
    #truncated_threshold = 3 / 2 * threshold  # wider threshold

    # get homogenous coordinates
    hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=dev)), dim=-1)
    hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=dev)), dim=-1)

    # calculate the sampson distance and msac scores
    M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))  # n_models, 3, n_pts
    M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))  # n_models, 3, n_pts
    JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
    x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)  # n_models, n_pts
    squared_distances = x1_M_x2_.square().div(JJ_T_)
    return squared_distances

def refine_poselib(points, models_init, K1, K2, weights, loss_type, loss_scale):

    cp = CameraPose()
    idcam = {
        "model": "PINHOLE",
        "width": 2,
        "height": 2,
        "params": [1., 1., 0., 0.]
    }
    f1 = (K1[0, 0] + K1[1, 1]) / 2
    f2 = (K2[0, 0] + K2[1, 1]) / 2
    bundle_opts = {'gradient_tol': 1e-7,
                   'step_tol': 1e-7,
                   'max_iterations': 20,
                   'loss_type': loss_type,
                   'loss_scale': loss_scale.item()}

    weights = weights.cpu().numpy()

    R, t = models_init[:, :3, :3], models_init[:, :3, -1]
    R, t = R.cpu().numpy(), t.cpu().numpy()
    points = points.cpu().numpy()
    models_out = []
    for i in range(len(R)):
        cp.R = R[i]
        cp.t = t[i]
        mask = weights[i] > 0
        pts = points[mask]
        w = list(np.square(weights[i][mask]))
        if loss_type == 'CAUCHY':
            pose, stats = refine_relative_pose(pts[:, :2], pts[:, 2:], cp,
                                               idcam, idcam, bundle_opts, w)
        else:
            pose, stats = refine_relative_pose(pts[:, :2], pts[:, 2:], cp,
                                               idcam, idcam, bundle_opts)
        models_out.append(np.concatenate([pose.R, pose.t[:, None]], axis=-1))
    return torch.tensor(np.stack(models_out, axis=0), dtype=models_init.dtype, device=models_init.device)

def refine_poselib_fmat(points, models_init, weights, loss_type, loss_scale):

    dv, dt = points.device, points.dtype
    bundle_opts = {'gradient_tol': 1e-9,
                   'step_tol': 1e-9,
                   'max_iterations': 100,
                   'loss_type': loss_type,
                   'loss_scale': loss_scale.item()}
    # msac = MSACBaseScoringModule(threshold=bundle_opts['loss_scale'] * np.sqrt(5.0), trainable=False)
    # wm = msac(models_init, points, K1, K2)
    # wm = wm.cpu().numpy()
    weights = weights.cpu().numpy()

    models_init = models_init.cpu().numpy()
    points = points.cpu().numpy()
    models_out = []
    for i in range(len(models_init)):
        mask = weights[i] > 0
        pts = points[mask]
        w = list(np.square(weights[i][mask]))
        if loss_type == 'CAUCHY':
            fmat, stats = refine_fundamental(pts[:, :2], pts[:, 2:], models_init[i],
                                               bundle_opts, w)
        else:
            fmat, stats = refine_fundamental(pts[:, :2], pts[:, 2:], models_init[i],
                                               bundle_opts)
        models_out.append(fmat)
    return torch.tensor(np.stack(models_out, axis=0), dtype=dt, device=dv)


class RelposeRefinerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_shape = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32) * 1e-1, requires_grad=False)
        self.prob_pow = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=True)

    def forward(self, points, models, K1, K2, prob, refinement_weights=None, loss_type='CAUCHY', *args, **kwargs):
        n_pts, _ = points.shape
        assert points.shape == (n_pts, 4)
        n_models, _, _ = models.shape
        assert models.shape == (n_models, 3, 4)

        f1 = (K1[0, 0] + K1[1, 1]) / 2
        f2 = (K2[0, 0] + K2[1, 1]) / 2
        loss_scale = (1/f1 + 1/f2) / 2

        z1z2 = triangulate_points_and_models(points, models)
        R, t = models[..., :3], models[..., -1]

        # Zero the weight of points whose chirality is wrong
        chirality = torch.all(z1z2 > 0.01, dim=-1)

        if loss_type == 'TRUNCATED':
            sqr_dist = compute_squared_distances(models, points)
            weights = (chirality & (sqr_dist < loss_scale.square() * 5)).float()
        elif loss_type == 'BARRON' or loss_type == 'CAUCHY':
            weights = chirality

            refinement_ablation_mode = False
            if refinement_ablation_mode:
                sqr_dist = compute_squared_distances(models, points)
                weights = (chirality & (sqr_dist < loss_scale.square())).float()
            elif refinement_weights is None:
                weights = weights.float() * prob.relu().detach() ** self.prob_pow
                weights[weights < 1e-3] = -1.0
            else:
                weights = weights.float() * refinement_weights

        n_nonzero = (weights > 0).float().sum(dim=-1)
        can_refine_mask = n_nonzero >= 6
        can_refine, = torch.where(can_refine_mask)
        if len(can_refine) < n_models:
            return models
        opt_dev = 'cpu'

        # Run custom sparse LM on torch, which allows gradient computation
        R_init = R[can_refine].to(opt_dev).double()
        t_init = t[can_refine].to(opt_dev).double()
        w = weights[can_refine].to(opt_dev).double()
        batch_index, data_index = torch.where(w > 0)
        w_in = w[batch_index, data_index]
        p_in = points[data_index].to(opt_dev).double()
        pts1, pts2 = p_in[:, :2], p_in[:, 2:]
        hom_pts1 = torch.nn.functional.pad(pts1, (0, 1), value=1.0)
        hom_pts2 = torch.nn.functional.pad(pts2, (0, 1), value=1.0)
        if loss_scale.ndim > 0:
            loss_scale_ = loss_scale[data_index]
        else:
            loss_scale_ = loss_scale
        loss_scale_ = loss_scale_.to(opt_dev).double()
        loss_scale = loss_scale.to(opt_dev).double()
        loss_shape = self.loss_shape.to(opt_dev).double()

        if loss_type == 'BARRON' or loss_scale.ndim > 0 or weights.requires_grad or models.requires_grad or loss_scale_.requires_grad or loss_shape.requires_grad:
            R_new, t_new = sparse_lm_loop(R_init, t_init, hom_pts1, hom_pts2, w_in, batch_index, loss_scale_, loss_shape, loss_type, 20, 1e-6)
            models_new = torch.cat([R_new, t_new.unsqueeze(-1)], dim=-1)
        else:
            # Run C++ refinement without gradients, same result but ~8x faster
            models_new = refine_poselib(points.to(opt_dev).double(), models[can_refine].to(opt_dev).double(), K1, K2, weights.to(opt_dev).double(),
                                        loss_type=loss_type, loss_scale=loss_scale.double())

        models_new = models_new.float().to(points.device)
        torch.set_default_tensor_type(torch.FloatTensor)

        return models_new

class FundamentalRefinerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_shape = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32) * 1e-1, requires_grad=False)
        self.prob_pow = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=True)

    def forward(self, points, models, prob, refinement_weights=None, loss_type='CAUCHY', *args, **kwargs):
        n_pts, _ = points.shape
        assert points.shape == (n_pts, 4)
        n_models, _, _ = models.shape
        assert models.shape == (n_models, 3, 3)

        loss_scale = torch.tensor(1.0, device=points.device, dtype=points.dtype)

        # Zero the weight of points whose chirality is wrong
        chirality = ~check_cheirality_fmat(models, points)

        if loss_type == 'TRUNCATED':
            sqr_dist = compute_squared_distances_fmat(models, points)
            weights = ((sqr_dist < 20)).float()
        elif loss_type == 'BARRON' or loss_type == 'CAUCHY':
            weights = chirality

            refinement_ablation_mode = False
            if refinement_ablation_mode:
                sqr_dist = compute_squared_distances_fmat(models, points)
                weights = (chirality & (sqr_dist < loss_scale.square())).float()
            elif refinement_weights is None:
                weights = weights.float() * prob.relu().detach() ** self.prob_pow
            else:
                weights = weights.float() * refinement_weights

            loss_scale = loss_scale # * self.loss_scale / prob.div(1-prob).sqrt()

        n_nonzero = (weights > 0).float().sum(dim=-1)
        can_refine_mask = n_nonzero >= 8
        can_refine, = torch.where(can_refine_mask)
        if len(can_refine) < n_models:
            return models
        opt_dev = 'cpu'

        # Run custom sparse LM on torch, which allows gradient computation
        models_init = models[can_refine].to(opt_dev).double()
        w = weights[can_refine].to(opt_dev).double()
        batch_index, data_index = torch.where(w > 0)
        w_in = w[batch_index, data_index]
        p_in = points[data_index].to(opt_dev).double()
        pts1, pts2 = p_in[:, :2], p_in[:, 2:]
        hom_pts1 = torch.nn.functional.pad(pts1, (0, 1), value=1.0)
        hom_pts2 = torch.nn.functional.pad(pts2, (0, 1), value=1.0)
        if loss_scale.ndim > 0:
            loss_scale_ = loss_scale[data_index]
        else:
            loss_scale_ = loss_scale
        loss_scale_ = loss_scale_.to(opt_dev).double()
        loss_scale = loss_scale.to(opt_dev).double()
        loss_shape = self.loss_shape.to(opt_dev).double()

        models_new = sparse_lm_loop_fmat(models_init, hom_pts1, hom_pts2, w_in, batch_index, loss_scale_, loss_shape, loss_type, 20, 1e-6)

        models_new = models_new.float().to(points.device)

        torch.set_default_tensor_type(torch.FloatTensor)
        return models_new


