import random
import pytorch_lightning as pl
import torch
import numpy as np


class CARANSACTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.map_thresholds = torch.tensor([1., 5., 10., 20.])

    def log_losses(self, log_dict, base_name):
        for k, v in log_dict.items():
            if v is not None:
                self.log(f"{base_name}/{k}", v.item(), batch_size=1, on_step=False, on_epoch=True)

    def compute_aucs(self, rte):
        aucs = (1. - rte.unsqueeze(-1).div(self.map_thresholds)).relu().mean(dim=0)
        aucs_dict = {f"auc{thr.item():.01f}": a for thr, a in zip(self.map_thresholds, aucs)}
        return aucs_dict

    def compute_maps(self, rte):
        maps = (rte.unsqueeze(-1) < self.map_thresholds).float().mean(dim=0)
        maps_dict = {f"map{thr.item():.01f}": m for thr, m in zip(self.map_thresholds, maps)}
        return maps_dict

    def compute_inlier_metrics(self, state):
        inl_hat = state["prob"]
        inl_gt = state["mask_gt"]

        inl_hat_hard = inl_hat > 0.5
        tot_samples = len(inl_gt)
        tot_pos = max(inl_gt.long().sum(), 1)
        TP = (inl_hat_hard & inl_gt).long().sum()
        TN = (~inl_hat_hard & ~inl_gt).long().sum()

        accuracy = (TP+TN)/tot_samples
        rec = TP / tot_pos
        prec = TP / (inl_hat_hard.long().sum() + 1e-8)
        fscore = 2 * (prec * rec) / (prec + rec + 1e-8)
        return {'accuracy': accuracy,
                'precision': prec,
                'recall': rec,
                'fscore': fscore}

    def derivative_metrics(self, state):
        return {**self.compute_inlier_metrics(state),
                **self.compute_maps(state["pose_error"]),
                **self.compute_aucs(state["pose_error"])}

    def gather_log_metrics(self, states):
        metrics = {}
        for i, s in enumerate(states):
            metrics = {**metrics,
                       **{f"iter{i}_{k}": v for (k, v) in self.derivative_metrics(s).items()},
                       f"iter{i}_pose_error": s["pose_error"],
                       f"iter{i}_inlier_xe": s["class_loss"]}
        return metrics


    def forward_with_losses(self, batch):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        points = batch['points']
        sides = batch['sides']

        # Set optionals
        if 'RT_gt' not in batch:
            # Triggers self supervision (use the final estimated RT as ground truth)
            batch['RT_gt'] = None
        if 'mask' not in batch:
            # Triggers using the ground truth RT to label correspondences
            batch['mask'] = None
        if 'K1K2' not in batch:
            # Fill in placeholders, needed for E but not needed for F estimation
            batch['K1K2'] = None, None

        K1, K2 = batch['K1K2']

        iter_states, loss = self.model.forward_with_loss({'points': points, 'sides': sides, 'K1': K1, 'K2': K2}, RT_gt=batch['RT_gt'], mask_gt=batch['mask'])

        return iter_states, loss

    def training_step(self, batch, batch_idx):
        states, loss = self.forward_with_losses(batch)
        self.log_losses({**self.gather_log_metrics(states), "loss": loss}, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        states, loss = self.forward_with_losses(batch)
        self.log_losses({**self.gather_log_metrics(states), "loss": loss}, "val")
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scd = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                         factor=0.5, patience=10,
                                                         threshold=0.0001, threshold_mode='rel',
                                                         cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        return {"optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scd,
                    "monitor": "train/loss"
                }
               }
