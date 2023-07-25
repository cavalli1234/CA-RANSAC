import torch

class PSampler_Auto:
    def __init__(self, n_samples_per_iter, min_corr=15, p_succ=0.9):
        # Store the number of iterations and samples per iteration
        self.n_samples_per_iter = n_samples_per_iter

        self.min_corr = min_corr
        self.p_succ = p_succ

    def sample(self, n_iters, prob, *args, **kwargs):

        # Given the relationship between the inlier rate and the probability 
        #    to sample at least one successful minimal sample:
        #
        # p_success = 1. - (1 - inlier_prob ** n_samples_per_iter) ** n_iters
        # 
        # we wish to control the sample pool's estimated inlier rate to guarantee a given p_success:
        min_prob = (1. - (1. - self.p_succ) ** (1./n_iters)) ** (1./self.n_samples_per_iter)

        index, = torch.where(prob > min_prob)
        n_selected = len(index)
        if n_selected < self.min_corr:
            _, index = torch.topk(prob, k=self.min_corr, sorted=False)
            n_selected = self.min_corr

        samples = torch.rand(n_iters, n_selected, device=prob.device, dtype=prob.dtype).log().topk(k=self.n_samples_per_iter).indices

        return index[samples]
