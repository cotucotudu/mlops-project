import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102


def download_data(data_dir: str = "data"):
    """
    Loads Oxford Flowers 102 using torchvision and saves to data_dir.
    Loads train, validation, test datasets.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Скачиваем splits (train, val, test)
    Flowers102(root=data_dir, split="train", download=True)
    Flowers102(root=data_dir, split="val", download=True)
    Flowers102(root=data_dir, split="test", download=True)
    print(f"Данные успешно скачаны и сохранены в папку '{data_dir}'.")


class FlowerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, resize, crop_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.crop_size = crop_size

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.RandomResizedCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.val_test_transforms = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = Flowers102(
            root=self.data_dir,
            split="train",
            transform=self.train_transforms,
            download=False,
        )
        self.val_dataset = Flowers102(
            root=self.data_dir,
            split="val",
            transform=self.val_test_transforms,
            download=False,
        )
        self.test_dataset = Flowers102(
            root=self.data_dir,
            split="test",
            transform=self.val_test_transforms,
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def main(cfg: DictConfig):
    download_data(cfg.data.raw_dir)

    datamodule = FlowerDataModule(
        data_dir=cfg.data.raw_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        resize=cfg.data.resize,
        crop_size=cfg.data.crop_size,
    )
    datamodule.setup()
    print(f"Train dataset size: {len(datamodule.train_dataset)}")
    print(f"Validation dataset size: {len(datamodule.val_dataset)}")
    print(f"Test dataset size: {len(datamodule.test_dataset)}")


if __name__ == "__main__":
    main()
