import numpy as np
import numba as nb
import torch

def rt_error(R_gt, T_gt, R_est, T_est):
    if not isinstance(R_est, torch.Tensor):
        R_est = torch.tensor(np.array(R_est))
    if not isinstance(T_est, torch.Tensor):
        T_est = torch.tensor(np.array(T_est))
    if not isinstance(R_gt, torch.Tensor):
        R_gt = torch.tensor(R_gt)
    if not isinstance(T_gt, torch.Tensor):
        T_gt = torch.tensor(T_gt)
    assert R_est.ndim == T_est.ndim + 1
    assert R_gt.ndim == T_gt.ndim + 1
    if R_est.ndim == 2:
        R_est = R_est.unsqueeze(0)
        T_est = T_est.unsqueeze(0)
    if R_gt.ndim == 2:
        R_gt = R_gt.unsqueeze(0)
        T_gt = T_gt.unsqueeze(0)
    assert R_est.ndim == 3
    assert R_gt.ndim == 3
    # P2 = R2 @ (R1T @ (x - t1) + t2 = R2 @ R1T @ x - R2 @ R1T @ t1 + t2
    # P2 = R @ (x - t) = Rx - Rt
    T_gt = T_gt.div(T_gt.norm(dim=-1, keepdim=True))
    T_est = T_est.div(T_est.norm(dim=-1, keepdim=True) + 1e-9)

    eps = 1e-6

    r_err = (torch.diagonal(R_est @ R_gt.transpose(-1, -2), dim1=-1, dim2=-2).sum(dim=-1)-1).div(2.).clamp(-1.+eps, 1.-eps).acos() / np.pi * 180
    t_err = (T_gt * T_est).sum(dim=-1).clamp(-1.+eps, 1.-eps).acos() / np.pi * 180

    return r_err, t_err

def compute_F(R, T, K1inv, K2inv):
    F = (K2inv.T @ cross_mat(T/np.linalg.norm(T)) @ R @ K1inv)
    return F

def normal_space(m):
    d1, d2 = m.shape[-2:]
    assert d1 > d2
    return torch.linalg.qr(m, mode='complete')[0][..., -(d1-d2):]

@torch.jit.script
def batch_2x2_inv(m, check_dets: bool=False):
    a = m[..., 0, 0]
    b = m[..., 0, 1]
    c = m[..., 1, 0]
    d = m[..., 1, 1]
    minv = torch.empty_like(m)
    det = a * d - b * c
    if check_dets:
        det[torch.abs(det) < 1e-10] = 1e-10
    minv[..., 0, 0] = d
    minv[..., 1, 1] = a
    minv[..., 0, 1] = -b
    minv[..., 1, 0] = -c
    return minv / det.unsqueeze(-1).unsqueeze(-1)


@torch.jit.script
def cross_mat(vector):
    # assert that the input has the correct shape
    assert vector.shape[-1] == 3

    # unpack the input tensor along the last dimension
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]

    # compute the cross product matrix using a single tensor operation
    return torch.stack([
        torch.zeros_like(x), -z, y,
        z, torch.zeros_like(x), -x,
        -y, x, torch.zeros_like(x)
    ], dim=-1).view(vector.size() + (3,))

def RT_to_E(rt_models):
    R, t = rt_models[..., :3], rt_models[..., -1]
    E = cross_mat(t.div(t.norm(dim=-1, keepdim=True))) @ R
    return E

@nb.njit
def qvec2rotmat(qvec):
    """
        Transform a quaternion to a rotation matrix.
        Assumes the quaternion to be already unit norm, in the form: [qw, qx, qy, qz]
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

@nb.njit
def randquat():
    u = np.random.rand()
    v = np.random.rand() * 2 * np.pi
    w = np.random.rand() * 2 * np.pi
    sqrtu = np.sqrt(u)
    sqrt1mu = np.sqrt(1-u)
    return np.array([sqrt1mu * np.sin(v), sqrt1mu * np.cos(v), sqrtu * np.sin(w), sqrtu * np.cos(w)])

def batch_3x3_det(matrix):
    # Check that the input is a 3x3 matrix
    assert matrix.shape[-2:] == (3, 3)
    # Compute the determinant using the formula for 3x3 matrices
    det = matrix[..., 0, 0] * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1]) \
          - matrix[..., 0, 1] * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0]) \
          + matrix[..., 0, 2] * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    return det
