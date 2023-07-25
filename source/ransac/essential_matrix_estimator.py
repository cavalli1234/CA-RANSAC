import torch
from poselib import relpose_5pt, essential_matrix_5pt
from functools import wraps
from utils import cross_mat, batch_2x2_inv
import numpy as np


def poselib_relpose_5pt_(pts):
    bsize, num, ch = pts.shape
    assert ch == 4
    assert num == 5
    points = pts.cpu().numpy().astype(np.float64)
    p1, p2 = points[..., :2], points[..., 2:]
    p1 = np.pad(p1, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.)
    p2 = np.pad(p2, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.)

    out = []
    out_idx = []
    for i, (p1_, p2_) in enumerate(zip(p1, p2)):
        pose = relpose_5pt(p1_, p2_)
        for p in pose:
            out.append(np.concatenate([p.R, p.t[:, None]], axis=-1))
            out_idx.append(i)
    return (torch.tensor(np.stack(out, axis=0), dtype=pts.dtype, device=pts.device),
            torch.tensor(out_idx, dtype=torch.long, device=pts.device))

def poselib_relpose_5pt(pts):
    bsize, num, ch = pts.shape
    assert ch == 4
    assert num == 5
    points = pts.cpu().numpy().astype(np.float64)
    p1, p2 = points[..., :2], points[..., 2:]
    p1 = np.pad(p1, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.)
    p2 = np.pad(p2, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.)
    p1 = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)
    p2 = p2 / np.linalg.norm(p2, axis=-1, keepdims=True)

    out = []
    out_idx = []
    for i, (p1_, p2_) in enumerate(zip(p1, p2)):
        Es = essential_matrix_5pt(p1_, p2_)
        out.extend(Es)
        out_idx.extend([i for _ in Es])
    E_models = torch.tensor(np.stack(out, axis=0), dtype=pts.dtype, device=pts.device)
    minimal_source = torch.tensor(out_idx, dtype=torch.long, device=pts.device)
    R, t, z1z2, origin_index = decompose_and_triangulate_minimal(pts[minimal_source], E_models)
    RT_models = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    out = RT_models, minimal_source[origin_index]
    return out

def dummy_relpose(pts):
    bsize, num, ch = pts.shape
    out = torch.zeros((bsize, 3, 4), dtype=pts.dtype, device=pts.device)
    out[:, :3, :3] = torch.eye(3)
    out[:, 0, -1] = 1.
    return out, None

class EssentialMatrixEstimator(object):

    def __init__(self):
        self.sample_size = 5

    def estimate_model(self, matches, weights = None):
        # minimal solver
        if matches.shape[1] == self.sample_size:
            return self.estimate_minimal_model(matches, weights)
        # non-minimal solver
        elif matches.shape[1] > self.sample_size:
            return self.estimate_minimal_model(matches, weights)
        else:
            raise ValueError(f"Cannot estimate a model from matches with shape {matches.shape}")
        return None

    def estimate_minimal_model(self, pts, weights=None):  # x1 y1 x2 y2
        """
            using 5 points to estimate Essential matrix.
        """
        if weights is None and not pts.requires_grad:
            return poselib_relpose_5pt(pts)
        batch_size, num, _ = pts.shape
        dev = pts.device
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]

        # get the points
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        # Step1: construct the A matrix, A F = 0.
        # 5 equations for 9 variables, A is 5x9 matrix containing epipolar constraints
        # Essential matrix is a linear combination of the 4 vectors spanning the null space of A
        a_59 = torch.ones(x1.shape, device=dev)
        A_s = torch.stack(
            (torch.mul(x1, x2), torch.mul(x1, y2), x1,
             torch.mul(y1, x2), torch.mul(y1, y2), y1,
             x2, y2, a_59), dim=-1)
        if weights is not None:
            # weights = weights.div(weights.mean(dim=1, keepdim=True))
            A_s = A_s * weights.unsqueeze(-1)

        try:
            # us, ds, vs = torch.linalg.svd(A_s)#, full_matrices=False) #A_s # eigenvalues in increasing order
            d, v = torch.linalg.eigh(A_s.transpose(-1, -2) @ A_s)
            v = v.flip(2).transpose(-1, -2)
        except:
            print(f"Error encountered in torch.linalg.svd on matrix {A_s}. Dying.")
            raise

        if A_s.shape[1] >= 8:
            # If the sample is non-minimal, run the 8pt algorithm
            null_space = v[:, -1 :].float().reshape(-1, 3, 3).transpose(-1, -2)
            # E_models = null_space
            # u2, d2, v2 = torch.linalg.svd(null_space)
            # E_models = u2 @ torch.diag(torch.tensor([1., 1., 0.], device=u2.device, dtype=u2.dtype)) @ v2
            E_models = null_space

            minimal_source = torch.arange(batch_size, device=pts.device).flatten()

            R, t, z1z2 = decompose_and_triangulate(pts.squeeze(), E_models, weights)
            RT_models = torch.concatenate([R, t.unsqueeze(-1)], dim=-1)
            return RT_models, minimal_source

        null_space = v[:, -4:, :].transpose(-1, -2).float()  # the last four rows

        # use the 4 eigenvectors according to the 4 smallest singular values,
        # E is calculated from 4 basis, E = cx*X + cy*Y + cz*Z + cw*W, up to common scale = 1
        null_space_mat = null_space.reshape(null_space.shape[0], 3, 3, null_space.shape[-1]).transpose(1,
                                                                                                       2)  # X, Y, Z, W

        # Step2: expansion of the constraints:
        # determinant constraint det(E) = 0,
        # trace constraint 2EE^TE - trace(EE^T)E = 0
        constraint_mat = get_constraint_mat(null_space_mat.cpu()).to(null_space_mat.device)

        # Step 3: Eliminate part of the matrix to isolate polynomials in z.
        # solve AX=b
        b = constraint_mat[:, :, 10:]
        eliminated_mat = torch.linalg.solve(constraint_mat[:, :, :10], b)

        action_mat = torch.zeros((batch_size, 10, 10), device=dev)
        eliminated_mat_idx = torch.tensor([0, 1, 2, 4, 5, 7], device=dev)
        action_mat[:, :6] = eliminated_mat[:, eliminated_mat_idx]
        action_mat[:, 6, 0] = -1
        action_mat[:, 7, 1] = -1
        action_mat[:, 8, 3] = -1
        action_mat[:, 9, 6] = -1

        ee, vv = torch.linalg.eig(action_mat)#, eigenvectors=True)# svd

        # put the cx, cy, cz back to get a valid essential matrix
        E_models = null_space.matmul(vv.real[:, -4:])#torch.stack(#.real
        real_mask = (vv.imag[:, -4:].abs() < 1e-3).all(dim=1).flatten()
        E_models = E_models.transpose(-1, -2).reshape(-1, 3, 3).transpose(-1, -2)
        minimal_source = torch.arange(batch_size, device=pts.device).unsqueeze(1).expand(-1, 10).flatten()

        E_models, minimal_source = E_models[real_mask], minimal_source[real_mask]

        R, t, z1z2, origin_index = decompose_and_triangulate_minimal(pts[minimal_source], E_models)
        RT_models = torch.concatenate([R, t.unsqueeze(-1)], dim=-1)
        out = RT_models, minimal_source[origin_index]

        return out

def trace_deco(*args, **kwargs):
    def trace_func(func):
        return wraps(func)(torch.jit.trace(func, *args, **kwargs))
    return trace_func

def get_constraint_mat(null_space):
    """expansion of the constraints.
    10*20"""
    # 1st: trace constraint 2EE^TE - trace(EE^T)E = 0
    # compute the EE^T

    batch_poly_mult = multiply_deg_one_poly(null_space.unsqueeze(2), null_space.unsqueeze(1))
    EE_t = 2 * batch_poly_mult.sum(dim=3)

    # trace
    trace = EE_t[:, 0, 0] + EE_t[:, 1, 1] + EE_t[:, 2, 2]

    batch_poly_mult = multiply_two_deg_one_poly(EE_t.unsqueeze(2), null_space.transpose(-2, -3).unsqueeze(1))
    trace_constraint = batch_poly_mult.sum(dim=3) - 0.5 * multiply_two_deg_one_poly(trace.unsqueeze(1).unsqueeze(1), null_space)

    # 2nd: singularity constraint det(E) = 0
    det_constraint = multiply_two_deg_one_poly(
        multiply_deg_one_poly(null_space[:, 0, 1], null_space[:, 1, 2]) -
        multiply_deg_one_poly(null_space[:, 0, 2], null_space[:, 1, 1]), null_space[:, 2, 0]) + \
                     multiply_two_deg_one_poly(
                         multiply_deg_one_poly(null_space[:, 0, 2], null_space[:, 1, 0]) -
                         multiply_deg_one_poly(null_space[:, 0, 0], null_space[:, 1, 2]),
                         null_space[:, 2, 1]) + \
                     multiply_two_deg_one_poly(
                         multiply_deg_one_poly(null_space[:, 0, 0], null_space[:, 1, 1]) -
                         multiply_deg_one_poly(null_space[:, 0, 1], null_space[:, 1, 0]),
                         null_space[:, 2, 2])

    # construct the overall constraint 10*20
    constraint_mat = torch.nn.functional.pad(trace_constraint.view(-1, 9, 20), (0, 0, 0, 1), "constant", 0.0)
    constraint_mat[:, 9] = det_constraint

    return constraint_mat

def multiply_deg_one_poly(a, b):
    """
        from Graph-cut Ransac
        Multiply two degree one polynomials of variables x, y, z.
        E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
        Output order: x^2 xy y^2 xz yz z^2 x y z 1 ('GrevLex', Graded reverse lexicographic order)
        1*10
        """

    return torch.stack([a[..., 0] * b[..., 0], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
                        a[..., 1] * b[..., 1], a[..., 0] * b[..., 2] + a[..., 2] * b[..., 0],
                        a[..., 1] * b[..., 2] + a[..., 2] * b[..., 1], a[..., 2] * b[..., 2],
                        a[..., 0] * b[..., 3] + a[..., 3] * b[..., 0], a[..., 1] * b[..., 3] + a[..., 3] * b[..., 1],
                        a[..., 2] * b[..., 3] + a[..., 3] * b[..., 2], a[..., 3] * b[..., 3]], dim=-1)


def multiply_deg_one_poly_alt(a, b):
    """
    from Graph-cut Ransac
    Multiply two degree one polynomials of variables x, y, z.
    E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
    Output order: x^2 xy y^2 xz yz z^2 x y z 1 ('GrevLex', Graded reverse lexicographic order)
    1*10
    """
    # \ b b b b
    # a 1 2 4 7
    # a 2 3 5 8
    # a 4 5 6 9
    # a 7 8 9 A

    idxa = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], device=a.device)
    idxb = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], device=a.device)
    out_prod = a.unsqueeze(-1) * b.unsqueeze(-2)
    coefficient_mat = out_prod + out_prod.transpose(-1, -2) * (1. - torch.eye(4, dtype=a.dtype, device=a.device))
    return coefficient_mat[:, idxa, idxb]

def multiply_two_deg_one_poly(a, b):
    """
    from Graph-cut Ransac
    Multiply two degree one polynomials of variables x, y, z.
    E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
    Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
    1*20
    """
    # x x 0 1 2 3
    # x \ b b b b
    # 0 a 1 2 5 B
    # 1 a 2 3 6 C
    # 2 a 3 4 7 D
    # 3 a 5 6 8 E
    # 4 a 6 7 9 F
    # 5 a 8 9 A G
    # 6 a B C E H

    return torch.stack([

        a[..., 0] * b[..., 0], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0], a[..., 1] * b[..., 1] + a[..., 2] * b[..., 0],
        a[..., 2] * b[..., 1], a[..., 0] * b[..., 2] + a[..., 3] * b[..., 0],
        a[..., 1] * b[..., 2] + a[..., 3] * b[..., 1] + a[..., 4] * b[..., 0], a[..., 2] * b[..., 2] + a[..., 4] * b[..., 1],
        a[..., 3] * b[..., 2] + a[..., 5] * b[..., 0],
        a[..., 4] * b[..., 2] + a[..., 5] * b[..., 1], a[..., 5] * b[..., 2],
        a[..., 0] * b[..., 3] + a[..., 6] * b[..., 0], a[..., 1] * b[..., 3] + a[..., 6] * b[..., 1] + a[..., 7] * b[..., 0],
        a[..., 2] * b[..., 3] + a[..., 7] * b[..., 1],
        a[..., 3] * b[..., 3] + a[..., 6] * b[..., 2] + a[..., 8] * b[..., 0],
        a[..., 4] * b[..., 3] + a[..., 7] * b[..., 2] + a[..., 8] * b[..., 1],
        a[..., 5] * b[..., 3] + a[..., 8] * b[..., 2], a[..., 6] * b[..., 3] + a[..., 9] * b[..., 0],
        a[..., 7] * b[..., 3] + a[..., 9] * b[..., 1], a[..., 8] * b[..., 3] + a[..., 9] * b[..., 2],
        a[..., 9] * b[..., 3]

    ], dim=-1)

def decompose_and_triangulate_minimal(points, models):
    n_models, n_minimal, _ = points.shape
    # assert n_minimal == 5
    assert points.shape == (n_models, n_minimal, 4)
    n_models, _, _ = models.shape
    assert models.shape == (n_models, 3, 3)

    Y1 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=models.dtype, device=models.device)
    Y2 = Y1.T
    u, d, vt = torch.linalg.svd(models)
    t1 = u[..., -1]
    t2 = -t1
    R1 = u @ Y1 @ vt
    R2 = u @ Y2 @ vt
    R_mirror = ((torch.linalg.det(R1) > 0).long() * 2 - 1).view(-1, 1, 1)
    R1 = R1 * R_mirror
    R2 = R2 * R_mirror

    # def err(t, R):
    #     E_hat = cross_mat(t) @ R
    #     dist = torch.linalg.norm(models - (models * E_hat).sum().div(E_hat.square().sum()) * E_hat).square()
    #     return dist
    # e1, e2, e3, e4 = err(t1, R1), err(t2, R2), err(t1, R2), err(t2, R1)

    R_batch = torch.stack([R1, R2, R1, R2], dim=0).view(-1, 3, 3)
    T_batch = torch.stack([t1, t1, t2, t2], dim=0).view(-1, 3)
    u1, u2 = points[..., :2], points[..., -2:]

    # inline triangulation:
    c1 = -torch.nn.functional.pad(u1, (0 ,1), value=1.).repeat(4, 1, 1) @ R_batch.transpose(-1, -2) # (n_models * 4, n_minimal, 3)
    c2 = torch.nn.functional.pad(u2, (0 ,1), value=1.).repeat(4, 1, 1)  # (n_models * 4, n_minimal, 3)
    c1c2 = torch.stack([c1, c2], axis=-1)
    z1z2 = batch_2x2_inv(c1c2.transpose(-1, -2) @ c1c2) @ c1c2.transpose(-1, -2) @ T_batch.view(-1, 1, 3, 1)
    # z1z2 = z1z2.view(4, -1, n_minimal, 2)

    # chirality_num_pass = torch.all(z1z2 > 0, dim=-1).long().sum(dim=-1)  # (4, n_models)
    # best_model = torch.argmax(chirality_num_pass, dim=0)
    # model_range = torch.arange(n_models, device=models.device)
    # R = R_batch.view(4, n_models, 3, 3)[best_model, model_range].squeeze(0)
    # T = T_batch.view(4, n_models, 3)[best_model, model_range].squeeze(0)
    # z1z2_ = z1z2[best_model, model_range].squeeze(0)
    # for i in range(n_models):
    #     print(i)
    #     print(z1z2[:, i])
    #     print(chirality_num_pass[:, i])
    # return R, T, z1z2_
    z1z2 = z1z2.view(-1, n_minimal * 2)
    chirality_pass = torch.all(z1z2 > 0.01, dim=-1)  # (4 * num_models)
    origin_index = chirality_pass.view(4, -1).nonzero(as_tuple=True)[1]
    R = R_batch[chirality_pass]
    T = T_batch[chirality_pass]
    z1z2 = z1z2[chirality_pass].view(-1, n_minimal, 2)
    return R, T, z1z2, origin_index





def decompose_and_triangulate(points, models, weights=None):
    n_matches, _ = points.shape
    assert points.shape == (n_matches, 4)
    n_models, _, _ = models.shape
    assert models.shape == (n_models, 3, 3)

    Y1 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=models.dtype, device=models.device)
    Y2 = Y1.T
    u, d, vt = torch.linalg.svd(models)
    t1 = u[..., -1]
    t2 = -t1
    R1 = u @ Y1 @ vt
    R2 = u @ Y2 @ vt
    R_mirror = ((torch.linalg.det(R1) > 0).long() * 2 - 1).view(n_models, 1, 1)
    R1 = R1 * R_mirror
    R2 = R2 * R_mirror

    # def err(t, R):
    #     E_hat = cross_mat(t) @ R
    #     dist = torch.linalg.norm(models - (models * E_hat).sum().div(E_hat.square().sum()) * E_hat).square()
    #     return dist
    # e1, e2, e3, e4 = err(t1, R1), err(t2, R2), err(t1, R2), err(t2, R1)

    R_batch = torch.stack([R1, R2, R1, R2], dim=0).view(n_models * 4, 3, 3)
    T_batch = torch.stack([t1, t1, t2, t2], dim=0).view(n_models * 4, 3)
    u1, u2 = points[:, :2], points[:, -2:]

    # z1z2 = triangulate(u1, u2, R_batch, T_batch).view(4, n_models, n_matches, 2)
    # inline triangulation:
    c1 = -torch.nn.functional.pad(u1, (0 ,1), value=1.).view(1, -1, 3) @ R_batch.transpose(-1, -2) # (n_models * 4, n_matches, 3)
    c2 = torch.nn.functional.pad(u2, (0 ,1), value=1.).view(1, n_matches, 3).expand(n_models * 4, n_matches, 3)  # (n_models * 4, n_matches, 3)
    c1c2 = torch.stack([c1, c2], axis=-1)
    z1z2 = batch_2x2_inv(c1c2.transpose(-1, -2) @ c1c2) @ c1c2.transpose(-1, -2) @ T_batch.view(n_models * 4, 1, 3, 1)
    z1z2 = z1z2.view(4, n_models, n_matches, 2)

    chirality_num_pass = torch.all(z1z2 > 0.01, dim=-1).float()  # (4, n_models, n_matches)
    if weights is not None:
        chirality_num_pass = chirality_num_pass * weights.squeeze()
    chirality_num_pass = chirality_num_pass.sum(-1)  # (4, n_models)

    best_model = torch.argmax(chirality_num_pass, dim=0)
    model_range = torch.arange(n_models, device=models.device)
    R = R_batch.view(4, n_models, 3, 3)[best_model, model_range].squeeze(0)
    T = T_batch.view(4, n_models, 3)[best_model, model_range].squeeze(0)
    return R, T, z1z2[best_model, model_range].squeeze(0)

def triangulate_points_and_models(points, rt):
    R, T = rt[..., :3], rt[..., -1]
    u1, u2 = points[..., :2], points[..., 2:]
    return triangulate(u1, u2, R, T)

def triangulate(u1, u2, R, T):
    # Let (R, t) be the transform that goes from im1 to im2, then:
    # z2 x2 = z1 R @ x1 + t
    # t = [-R @ x1, x2] @ [z1; z2]
    # [z1; z2] ~= pinv([-R @ x1, x2]) @ t
    n_models = R.shape[0]
    n_pts = u1.shape[0]
    assert R.shape == (n_models, 3, 3)
    assert T.shape == (n_models, 3)
    assert u1.shape == (n_pts, 2)
    assert u2.shape == (n_pts, 2)
    c1 = -torch.nn.functional.pad(u1, (0 ,1), value=1.).view(1, -1, 3) @ R.transpose(-1, -2) # (n_models, n_pts, 3)
    c2 = torch.nn.functional.pad(u2, (0 ,1), value=1.).view(1, n_pts, 3).expand(n_models, n_pts, 3)  # (n_models, n_pts, 3)
    # z1z2 = torch.linalg.lstsq(torch.stack([c1, c2], axis=-1), T.view(n_models, 1, 3, 1)).solution.squeeze()
    c1c2 = torch.stack([c1, c2], axis=-1)
    z1z2 = batch_2x2_inv(c1c2.transpose(-1, -2) @ c1c2) @ c1c2.transpose(-1, -2) @ T.view(n_models, 1, 3, 1)
    return z1z2.squeeze(-1)

if __name__ == '__main__':
    import numpy as np
    import math
    from data.utils import rt_error

    from data.torch_datasets import OutlierRejectionDataset, PhototourismDataIndex

    ds = OutlierRejectionDataset(PhototourismDataIndex('sacre_coeur', 'val'))
    for batch in ds:
        E, rt, x = batch['E'], batch['RT'], batch['points']
        print("_"*10)
        R_est, T_est, _ = decompose_and_triangulate(x, E.unsqueeze(0))
        R_gt, T_gt = rt[:3, :3], rt[:3, -1]
        print(rt_error(R_gt, T_gt, R_est, T_est))
