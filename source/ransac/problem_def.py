from .essential_matrix_estimator import EssentialMatrixEstimator
from .fundamental_matrix_estimator import FundamentalMatrixEstimator
from .refinement import RelposeRefinerModule, FundamentalRefinerModule
from .scoring import FundamentalMSAC, EssentialMSAC


class ProblemDefinition:
    def __init__(self, n_minimal, datapoint_dim, solver, scorer, refiner=None):
        self.n_minimal = n_minimal
        self.datapoint_dim = datapoint_dim
        self.solver = solver
        self.scorer = scorer
        self.refiner = refiner


class EssentialMatrixEstimationProblem(ProblemDefinition):
    def __init__(self):
        super().__init__(5, 4, EssentialMatrixEstimator(), EssentialMSAC(), RelposeRefinerModule())

class FundamentalMatrixEstimationProblem(ProblemDefinition):
    def __init__(self):
        super().__init__(7, 4, FundamentalMatrixEstimator(), FundamentalMSAC(), FundamentalRefinerModule())

