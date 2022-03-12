import pytorch_lightning as pl
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
    model = AllergenClassifier()
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(
        model,
        DataLoader(
            train_dataset,
            num_workers=5,
            batch_size=8,
            shuffle=True,
            worker_init_fn=set_worker_sharing_strategy),
        DataLoader(
            val_dataset,
            num_workers=5,
            batch_size=8,
            worker_init_fn=set_worker_sharing_strategy)
        )


if __name__ == '__main__':
    main()
