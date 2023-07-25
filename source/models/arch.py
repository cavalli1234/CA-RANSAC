import numpy as np
import torch
from utils import rt_error
from torch import nn
from ransac.samplers import PSampler_Auto
from ransac.problem_def import EssentialMatrixEstimationProblem
from ransac.scoring import EssentialMSAC
from .components import FusionMLPFeatureUpdateModule, MLPProbabilityModel, SideMLPInitializer, CoordinateEmbedding

class CA_RANSAC(nn.Module):
    def __init__(self, f_dim=128, n_iterations=4, n_minimal_samples_per_iter=256, n_sides=1, nefsac_multiplier=None, problem_definition=None):
        super().__init__()

        self.n_sides = n_sides

        # Init module: initializes the features for each data point, given the data points
        self.init_module = SideMLPInitializer(dims=[n_sides, f_dim, f_dim], embedder=CoordinateEmbedding(num_frequencies=8, rng=1.5))

        # Probability from features: estimates the points' likelihood of being inlier from its features
        self.f2p_module = MLPProbabilityModel(f_dim, n_layers=3)

        self.nefsac_multiplier = nefsac_multiplier or 1
        self.nefsac = torch.jit.load("../resources/models/nefsac_E_phototourism.pt", map_location='cpu')

        # Updates features based on given attention weights. F' = FUM(F, A)
        self.feature_update_module = FusionMLPFeatureUpdateModule(f_dim, n_layers=3)

        # Defines the geometry of the problem to solve, includes a minimal solver
        self.problem_definition = problem_definition or EssentialMatrixEstimationProblem()

        # Number of iterations
        self.n_iterations = n_iterations

        # Number of new minimal samples to take at random on every iteration
        self.n_minimal_samples = n_minimal_samples_per_iter

        # Number of best models to consider for attention propagation on every iteration
        # The runtime cost of the attention mechanism is linear in the number of correspondences and in the number of models used
        self.n_top_models = 4

        # Number of best models to preserve for the next iteration
        self.n_bring_forward = 1

        # Sampler
        self.sampler = PSampler_Auto(self.problem_definition.n_minimal, min_corr=15, p_succ=0.9)

    def sample_minimal_models(self, x, n):
        model_hypotheses = []
        while len(model_hypotheses) == 0:
            minimal_sample_indices = self.sampler.sample(**x, n_iters=n * self.nefsac_multiplier, repeat=False)
            minimal_sample_coordinates = x['points'][minimal_sample_indices]
            if self.nefsac_multiplier > 1:
                nefsac_scores = self.nefsac(minimal_sample_coordinates).squeeze()
                best_samples_idx = torch.topk(nefsac_scores, n).indices
                minimal_sample_coordinates = minimal_sample_coordinates[best_samples_idx]
            model_hypotheses, minimal_source = self.problem_definition.solver.estimate_model(minimal_sample_coordinates)
        return model_hypotheses

    @staticmethod
    def print_state(s):
        print({k: v.shape for k, v in s.items()})

    def forward(self, x):
        """
            Main entry point for inference in CA-RANSAC.

            The input x is a dictionary holding all of the necessary values for the included modules to operate.
            The current implementation requires the following fields:
                x["points"]: a Nx4 tensor with the N correspondences in normalized camera coordinates (E estimation) or pixel coordinates (F estimation)
                x["sides"]: a NxC tensor with the N side information of dimensionality C each. C must be the same as self.n_sides
                x["K1"], x["K2"]: only accessed for E estimation, each holds the 3x3 intrinsics matrix of each image

            The output is a list of all estimation states.
            Out[0] is the initialization before any iteration is performed.
            Out[i] is the state at the end of the i-th batch of iterations after i feature propagation stages

            Each estimation state holds complete information about the estimation state over each iterations, most importantly:
                Out[i]["prob"]: a tensor holding inlier probabilities in [0, 1] for each correspondence
                Out[i]["best_models"]: a tensor containing self.n_bring_forward estimated models in decreasing order of scores.
                                       models are 3x3 for F estimation and 3x4 for E estimation (formatted as a 3x3 rotation followed by a 1x3 unit translation)

        """
        # Initialize the state
        x["features"] = self.init_module(**x)

        # Precompute initial sampling probabilities
        x["prob"] = self.f2p_module(**x)
        x["iterno"] = 0


        # Draw the first top models at random
        init_models = self.sample_minimal_models(x, self.n_top_models)[:self.n_top_models]

        # Init state of the first iteration
        it0_state = {**x, "models": init_models}

        it0_state["score_matrix"] = self.problem_definition.scorer(**it0_state, check_chierality=False)  # n_models, n_points
        init_model_scores = it0_state['score_matrix'].sum(1)  # n_models

        it0_state["models"] = it0_state["models"]

        it0_state["best_models"] = it0_state["models"][:self.n_bring_forward]
        it0_state["best_model_scores"] = init_model_scores[:self.n_bring_forward]

        # Perform iterations and record all iteration states
        states_record = [it0_state]
        for ni in range(self.n_iterations):
            states_record.append(self.perform_iteration(states_record[-1]))

        return states_record

    def compute_rt_error(self, RT_hat, RT_gt):
        if RT_hat.ndim == 2:
            RT_hat_ = RT_hat.unsqueeze(0)
        else:
            RT_hat_ = RT_hat
        R_hat, T_hat = RT_hat_[..., :3], RT_hat_[..., -1]
        rte = torch.maximum(*rt_error(RT_gt[:3, :3], RT_gt[:3, -1], R_hat, T_hat))
        return rte

    def forward_with_loss(self, x, RT_gt=None, mask_gt=None):
        """
            Utility function for training and benchmarking.
            Calls the forward with the given input x, and then computes pose error and inlier cross entropy.

            If the inlier mask (mask_gt) is not given, it uses the ground truth pose to determine inliers based on their sampson error.
            If the ground truth pose (RT_gt) is not given, it uses the final estimated model as ground truth for self-supervision.
                Note that we observed self supervision to be almost as effective as complete supervision on PhotoTourism,
                provided that the model is given a meaningful initialization, such as pre-training with a synthetic dataset.
                For self-supervision to make sense, it is important to have at least self.n_iterations > 1,
                since the last iteration's result is improving indirectly by supervising the previous iterations.
        """
        states = self.forward(x)
        if RT_gt is None:
            # If no ground truth is provided, then self supervise
            RT_gt = states[-1]['best_models'].squeeze()
        states = self.forward_with_metrics(x, RT_gt, mask_gt, states)
        for state in states:
            state['loss'] = state['class_loss'] + state['pose_error'].clamp(0., 30.) / 60.
        n_itr = len(states)
        loss_weights = 0.9 ** np.arange(n_itr)
        loss_weights = loss_weights / np.sum(loss_weights)
        loss = sum(states[n_itr-i-1]['loss'] * loss_weights[i] for i in range(n_itr))
        return states, loss


    def forward_with_metrics(self, x, RT_gt, mask_gt=None, states=None):
        if states is None:
            # Only run the estimation if not done already
            states = self.forward(x)
        if mask_gt is None:
            # If no mask is provided, then infer it from the ground truth pose
            # Note: if running on F matrix, please also provide the intrinsic matrices in x 
            #       for this step to work. Otherwise, provide F_gt and a FundamentalMSAC score.
            mask_gt = EssentialMSAC()(**x, models=RT_gt.unsqueeze(0), check_chierality=True).squeeze() > 1e-6

        for state in states:
            RT_error = self.compute_rt_error(state['best_models'], RT_gt)
            class_loss = nn.functional.binary_cross_entropy(state["prob"].clamp(1e-6, 1.0-1e-6), mask_gt.float())
            state["class_loss"] = class_loss
            state["mask_gt"] = mask_gt
            state["pose_error"] = RT_error

        return states


    def perform_iteration(self, s):
        # Make a local copy of the state to avoid overwriting previous references
        s = {**s}
        s["iterno"] = s["iterno"] + 1
        new_models = self.sample_minimal_models(s, self.n_minimal_samples)
        s["models"] = torch.cat([s["best_models"], new_models], dim=0)  # n_models, [model_dims]
        # s["models"][0] = s["rtgt"][:3, :4]
        s["score_matrix"] = self.problem_definition.scorer(**s, check_chierality=False)  # n_models, n_points
        model_scores = s['score_matrix'].sum(1)  # n_models

        if self.n_top_models < len(model_scores):
            # Plainly ignore the worst models and only keep the best k
            model_scores, topk_index = torch.topk(model_scores, self.n_top_models, largest=True, sorted=True)
            s["models"] = s["models"][topk_index]
            s["score_matrix"] = s["score_matrix"][topk_index]

        s["models"] = self.problem_definition.refiner(**s, loss_type='TRUNCATED')
        s["score_matrix"] = self.problem_definition.scorer(**s, check_chierality=True)  # n_models, n_points
        model_scores = s['score_matrix'].sum(1)  # n_models

        s['score_matrix'] = s['score_matrix'].div(model_scores.sum().sqrt())

        # Consensus adaptive feature update with linear attention
        s["features"] = self.feature_update_module(**s)

        # Update the inlier probability estimates
        s["prob"] = self.f2p_module(**s)

        # Keep models sorted by scoring, to easily filter which to bring forward
        sorting_idxing = torch.argsort(model_scores, descending=True)
        model_scores = model_scores[sorting_idxing]
        s["score_matrix"] = s["score_matrix"][sorting_idxing]
        s["models"] = s["models"][sorting_idxing]

        s["best_models"] = s["models"][:self.n_bring_forward]
        s["best_model_scores"] = model_scores[:self.n_bring_forward]

        refinement_state = {**s}
        refinement_state["models"] = refinement_state["best_models"]
        refinement_state["models"] = self.problem_definition.refiner(**refinement_state, loss_type='CAUCHY')
        s["best_models"] = refinement_state["models"]

        return s
