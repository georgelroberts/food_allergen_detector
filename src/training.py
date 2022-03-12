from tkinter import W
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.load_data import get_train_val_test, clean_dataset
from src.model import AllergenClassifier
import torch.multiprocessing
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(
        sharing_strategy)


def main():
    clean_dataset()
    train_dataset, val_dataset, test_dataset = get_train_val_test()
    wandb_logger = WandbLogger()
    model = AllergenClassifier()
    wandb_logger.watch(model, log="all")
    trainer = pl.Trainer(max_epochs=20, logger=wandb_logger)
    trainer.fit(
        model,
        DataLoader(
            train_dataset,
            num_workers=0,
            batch_size=32,
            shuffle=True,
            worker_init_fn=set_worker_sharing_strategy),
        DataLoader(
            val_dataset,
            num_workers=0,
            batch_size=32,
            worker_init_fn=set_worker_sharing_strategy)
        )


if __name__ == '__main__':
    main()
