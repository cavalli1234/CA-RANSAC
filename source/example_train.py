from trainer import CARANSACTrainer
from models.arch import CA_RANSAC
from data.torch_datasets import PoseEstimationDataset, PhototourismDataIndex, SyntheticDataIndex
import pytorch_lightning as pl
import torch


def setup_dataloaders(data_dir, dataset):
    if dataset == 'phototourism':
        train_ds = PoseEstimationDataset(PhototourismDataIndex(PhototourismDataIndex.TRAIN_LIST, data_dir), maxlen=len(PhototourismDataIndex.TRAIN_LIST) * 10, deterministic=False)
        val_ds = PoseEstimationDataset(PhototourismDataIndex(PhototourismDataIndex.VAL_LIST, data_dir), maxlen=len(PhototourismDataIndex.VAL_LIST) * 25, deterministic=True)
    elif dataset == 'synthetic':
        train_ds = PoseEstimationDataset(SyntheticDataIndex(1500), maxlen=1500, deterministic=False)
        val_ds = PoseEstimationDataset(SyntheticDataIndex(250), maxlen=250, deterministic=False)
    else:
        raise ValueError(f"Dataset {dataset} unknown.")

    return torch.utils.data.DataLoader(train_ds, batch_size=None), torch.utils.data.DataLoader(val_ds, batch_size=None)


def main(checkpoint_out: str=None, checkpoint_in: str=None, dataset: str='phototourism', data_dir: str=None, tensorboard_path="/tmp/caransac_log", max_epochs=100):
    model = CA_RANSAC()
    pl_model = CARANSACTrainer(model, lr=1e-3)

    callbacks = [pl.callbacks.progress.TQDMProgressBar(refresh_rate=25)]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(checkpoint_out, monitor='val/loss', mode='min', save_top_k=1)
    callbacks.append(checkpoint_callback)
    logger = pl.loggers.TensorBoardLogger(tensorboard_path)

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='cpu', accumulate_grad_batches=16, gradient_clip_val=1.0, gradient_clip_algorithm="value", callbacks=callbacks, log_every_n_steps=20, logger=logger)

    if checkpoint_in is not None:
        sd = torch.load(checkpoint_in)
        pl_model.load_state_dict(sd['state_dict'], strict=True)

    train_dl, val_dl = setup_dataloaders(data_dir, dataset)

    trainer.fit(pl_model, train_dl, val_dl)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
