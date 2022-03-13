from tkinter import W
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import BackboneFinetuning, ModelCheckpoint
from torch.utils.data import DataLoader
import torch.multiprocessing

from src.load_data import get_train_val_test
from src.model import AllergenClassifier
from src.model_callbacks import (
    PlotSamplesCallback,
    PlotIncorrectSamplesCallback
)
from src.file_locations import MODEL_CHECKPOINTS_DIR
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
    train_dataloader, val_dataloader = get_train_val_dataloaders()
    model = AllergenClassifier(learning_rate=4.786e-3)
    train_model(model, train_dataloader, val_dataloader)
    # find_best_lr(model, train_dataloader, val_dataloader)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(
        sharing_strategy)


def train_model(model, train_dataloader, val_dataloader):
    wandb_logger = WandbLogger()
    wandb_logger.watch(model, log="all")
    backbone_finetuning = BackboneFinetuning(
        unfreeze_backbone_at_epoch=20)
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINTS_DIR,
        filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
        monitor="val_accuracy",
        mode="max")
    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[
            backbone_finetuning,
            checkpoint_callback,
            PlotSamplesCallback(wandb_logger),
            PlotIncorrectSamplesCallback(wandb_logger, 25)])
    trainer.fit(model, train_dataloader, val_dataloader)


def get_train_val_dataloaders():
    train_dataset, val_dataset, _ = get_train_val_test()
    print(f"{len(train_dataset)} training samples")
    train_imbalance = sum(train_dataset.labels) / len(train_dataset) * 100
    print(f"Train imbalance: {train_imbalance:.2f}% with gluten")
    val_imbalance = sum(val_dataset.labels) / len(val_dataset) * 100
    print(f"Val imbalance: {val_imbalance:.2f}% with gluten")
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
        worker_init_fn=set_worker_sharing_strategy),
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=0,
        batch_size=64,
        worker_init_fn=set_worker_sharing_strategy)
    return train_dataloader, val_dataloader


def find_best_lr(model, train_dataloader, val_dataloader):
    trainer = pl.Trainer()
    lr_finder = trainer.tuner.lr_find(
        model, train_dataloader, val_dataloader)
    print(lr_finder.results)
    print(lr_finder.suggestion())
    import matplotlib.pyplot as plt
    fig = lr_finder.plot(suggest=True)
    fig.show()


if __name__ == '__main__':
    main()
