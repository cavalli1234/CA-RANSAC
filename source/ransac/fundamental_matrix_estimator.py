import torch
import math
import poselib
import numpy as np

class FundamentalMatrixEstimator(object):

    def __init__(self):
        self.sample_size = 7

    def estimate_model(self, matches, weights=None):
        if matches.shape[1] == self.sample_size:
             return self.estimate_minimal_model(matches, weights)
        elif matches.shape[1] > self.sample_size:
            normalized_matches, T1, T2t = self.normalize(matches)
            return self.estimate_non_minimal_model(normalized_matches, T1, T2t, weights)
        return None

    def normalize(self, matches):
        dev = matches.device
        # The number of points in each minimal sample
        num_points = matches.shape[1]
        # Calculate the mass point for each minimal sample
        mass = torch.mean(matches, dim=1)
        # Substract the mass point of each minimal sample from the corresponding points in both images
        matches = matches - torch.unsqueeze(mass, 1).repeat(1, num_points, 1)
        # Calculate the distances from the mass point for each minimal sample in the source image
        distances1 = torch.linalg.norm(matches[:, :, :2], dim=2)
        # Calculate the distances from the mass point for each minimal sample in the destination image
        distances2 = torch.linalg.norm(matches[:, :, 2:], dim=2)
        # Calculate the average distances in the source image
        avg_distance1 = torch.mean(distances1, dim=1)
        # Calculate the average distances in the destination image
        avg_distance2 = torch.mean(distances2, dim=1)
        # Calculate the scaling to make the average distances sqrt(2) in the source image
        ratio1 = math.sqrt(2) / avg_distance1
        # Calculate the scaling to make the average distances sqrt(2) in the destination image
        ratio2 = math.sqrt(2) / avg_distance2

        # Calculate the normalized matches in the source image
        normalized_matches1 = matches[:, :, :2] * ratio1.view(-1, 1, 1).repeat(1, num_points, 2)
        # Calculate the normalized matches in the destination image
        normalized_matches2 = matches[:, :, 2:] * ratio2.view(-1, 1, 1).repeat(1, num_points, 2)

        # Initialize the normalizing transformations for each minimal sample in the source image
        T1 = torch.zeros((matches.shape[0], 3, 3), device=dev)
        # Initialize the normalizing transformations for each minimal sample in the destination image
        T2 = torch.zeros((matches.shape[0], 3, 3), device=dev)

        # Calculate the transformation parameters
        T1[:, 0, 0] = T1[:, 1, 1] = ratio1[:]
        T2[:, 0, 0] = T2[:, 1, 1] = ratio2[:]
        T1[:, 2, 2] = T2[:, 2, 2] = 1
        T1[:, 0, 2] = -ratio1 * mass[:, 0]
        T1[:, 1, 2] = -ratio1 * mass[:, 1]
        T2[:, 2, 0] = -ratio2 * mass[:, 2]
        T2[:, 2, 1] = -ratio2 * mass[:, 3]

        return torch.cat((normalized_matches1, normalized_matches2), dim=2), T1, T2

    def coeff(self, f1, f2):
        # The coefficient calculation for the 7PT algorithm
        d1 = torch.linalg.det(f1)
        d2 = torch.linalg.det(f2)
        d3 = torch.linalg.det(-f1 + 2 * f2)
        d4 = torch.linalg.det(2 * f1 - f2)
        d5 = torch.linalg.det(-2 * f1 + 3 * f2)
        c0 = d2
        c1 = (d1 - d3) / 3 - (d4 - d5) / 12
        c2 = 0.5 * d1 + 0.5 * d3 - d2
        c3 = (d1 - d3) / 6 - (d4 - d5) / 12
        c = torch.stack((c0, c1, c2, c3), dim=-1)
        return c

    # Eight-point algorithm
    def estimate_non_minimal_model(self, pts, T1, T2t, weights=None):  # x1 y1 x2 y2
        """
        Using 8 points and singularity constraint to estimate Fundamental matrix.
        """
        dev = pts.device
        # get the points
        B, N, _ = pts.shape
        #matches, transform1, transform2 = self.normalize(pts)
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        a_89 = torch.ones(x1.shape, device=dev)

        # construct the A matrix, A F = 0. 8 equations for 9 variables, solution is linear subspace o dimensionality of 2.
        #print(weights)
        if weights is not None:
            A = weights.unsqueeze(-1) * torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_89), dim=-1)#weights.unsqueeze(-1) *
        else:
            A = torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_89), dim=-1)

        # solve null space of A to get F
        try:
            _, _, v = torch.linalg.svd(A.transpose(-1, -2)@A)#, full_matrices=False)  # eigenvalues in increasing order
        except:
            print()
        null_space = v[:, -1:, :].transpose(-1, -2).float().clone()  # the last four rows

        # with the singularity constraint, use the last two singular vectors as a basis of the space
        F = null_space[:, :, 0].view(-1, 3, 3)

        if T1 is not None:
            for i in range(F.shape[0]):
                F[i, :, :] = torch.mm(T2t[i, :, :], torch.mm(F[i, :, :].clone(), T1[i, :, :]))

        return F, None

    def estimate_minimal_model(self, pts, weights=None):  # x1 y1 x2 y2
        """
        using 7 points and singularity constraint to estimate Fundamental matrix.
        """

        dev = pts.device
        # get the points
        B, N, _ = pts.shape
        pts1 = pts[:, :, 0:2]
        pts2 = pts[:, :, 2:4]
        x1, y1 = pts1[:, :, 0], pts1[:, :, 1]
        x2, y2 = pts2[:, :, 0], pts2[:, :, 1]

        a_79 = torch.ones(x1.shape, device=dev)

        # construct the A matrix, A F = 0. 7 equations for 9 variables,
        # solution is linear subspace o dimensionality of 2.
        A = torch.stack((x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, a_79), dim=-1)

        # solve null space of A to get F
        _, _, v = torch.linalg.svd(A.transpose(-1, -2)@A)  # eigenvalues in increasing order
        null_space = v[:, -2:, :].transpose(-1, -2).float()  # the last two rows

        # with the singularity constraint, use the last two singular vectors as a basis of the space
        F1 = null_space[:, :, 0].view(-1, 1, 3, 3)
        F2 = null_space[:, :, 1].view(-1, 1, 3, 3)

        # use the two bases, we can have an arbitrary F mat
        # lambda, 1-lambda, det(F) = det(lambda*F1, (1-lambda)*F2) = lambda(F1-F2)+F2 to find lambda
        # c-polynomial coefficients. det(F) = c[0]*lambda^3 + c[1]*lambda^2  + c[2]*lambda + c[3]= 0
        c = self.coeff(F1, F2).squeeze()

        # Find the roots of the polynomial c by the companion matrix method
        compmat = torch.zeros((c.shape[0], 4, 4), dtype=c.dtype, device=dev)
        compmat[:, 1, 0] = 1.
        compmat[:, 2, 1] = 1.
        compmat[:, 3, 2] = 1.
        compmat[..., 2] = -c
        vv = torch.linalg.eigvals(compmat)

        roots = vv.real
        s = F1[..., 2, 2] * roots + F2[..., 2, 2]
        valid_mask = (s > 1e-10) & (vv.imag.abs() < 1e-9)
        mu = 1.0 / s
        lambda_ = roots * mu
        Fs = F1 * lambda_.view(-1, 4, 1, 1) + F2 * mu.view(-1, 4, 1, 1)
        Fs = Fs[valid_mask].view(-1, 3, 3)
        pl_models = []
        for p, F in zip(pts, Fs):
            ransac_opt = {
                'max_iterations': 1,
                'min_iterations': 1,
                'max_epipolar_error': 1000,
                'progressive_sampling': False
            }
            p1, p2 = p[:, :2].cpu().numpy(), p[:, 2:].cpu().numpy()
            model, stats = poselib.estimate_fundamental(p1, p2, ransac_opt)
            pl_models.append(model)
            inl = stats["inliers"]
        Fs = torch.tensor(np.stack(pl_models, axis=0), dtype=pts.dtype, device=pts.device)
        return Fs, None


def cheirality_check(models, points):
    epipoles = torch.linalg.cross(models[..., 0, :], models[..., 2, :])
    degenerate_mask = torch.linalg.norm(epipoles, dim=-1) < 1e-6
    if torch.any(degenerate_mask):
        epi2 = torch.linalg.cross(models[degenerate_mask][..., 1, :], models[degenerate_mask][..., 2, :])
        epipoles[degenerate_mask] = epi2
    s1 = models[..., 0, 0].unsqueeze(1) * points[..., 2] + models[..., 1, 0].unsqueeze(1) * points[..., 3] + models[..., 2, 0].unsqueeze(1)
    s2 = epipoles[..., 1].unsqueeze(1) - epipoles[..., 2].unsqueeze(1) * points[..., 1]
    s = s1 * s2 > 0

    not_ok = s.float().mean(dim=-1) > 0.5
    s[not_ok] = ~s[not_ok]
    return s
