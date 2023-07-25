from .essential_matrix_estimator import triangulate_points_and_models
import torch
from torch import nn
from utils import RT_to_E

class FundamentalMSAC(nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    @staticmethod
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

        # calculate the sampson distance and msac scores
        M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))  # n_models, 3, n_pts
        M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))  # n_models, 3, n_pts
        JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
        x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)  # n_models, n_pts
        squared_distances = x1_M_x2_.square().div(JJ_T_)
        return squared_distances

    def forward(self, models, points, check_chierality=False, *args, **kwargs):
        if self.threshold is None:
            thr = 1.0
        else:
            thr = self.threshold
        squared_distances = self.compute_squared_distances(models, points)
        if check_chierality:
            cheirality_fail_mask = check_cheirality_fmat(models, points)
            squared_distances[cheirality_fail_mask] = 1.
        return 1. - squared_distances.div(thr * thr).clamp(0., 1.)

class EssentialMSAC(nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def compute_squared_distances(models, points):
        pts1 = points[:, 0:2]
        pts2 = points[:, 2:4]
        dev = points.device
        assert dev == models.device

        num_pts = pts1.shape[0]

        # get homogenous coordinates
        hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=dev)), dim=-1)
        hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=dev)), dim=-1)

        # Here models are assumed to be in the form of rotation and translation
        models = RT_to_E(models)

        # calculate the sampson distance and msac scores
        M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))  # n_models, 3, n_pts
        M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))  # n_models, 3, n_pts
        JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2  # n_models, n_pts
        x1_M_x2_ = hom_pts1.T.unsqueeze(0).mul(M_x2_).sum(-2)  # n_models, n_pts
        squared_distances = x1_M_x2_.square().div(JJ_T_)
        return squared_distances

    def forward(self, models, points, K1, K2, check_chierality=False, *args, **kwargs):
        if self.threshold is None:
            f1 = (K1[0, 0] + K1[1, 1]) / 2
            f2 = (K2[0, 0] + K2[1, 1]) / 2
            thr = (1/f1 + 1/f2) / 2
        else:
            thr = self.threshold
        squared_distances = self.compute_squared_distances(models, points)
        if check_chierality:
            z1z2 = triangulate_points_and_models(points, models)
            chirality_fail_mask = (z1z2 < 0.01).any(dim=-1)
            squared_distances[chirality_fail_mask] = 1.
        return 1. - squared_distances.div(thr * thr).clamp(0., 1.)

