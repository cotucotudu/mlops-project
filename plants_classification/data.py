import os
from torchvision.datasets import Flowers102
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def download_data(data_dir: str = "data"):
    """
    Загружает датасет Oxford Flowers 102 через torchvision и сохраняет в data_dir.
    Вызывает скачивание тренировочного, валидационного и тестового наборов.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Скачиваем splits (train, val, test)
    Flowers102(root=data_dir, split='train', download=True)
    Flowers102(root=data_dir, split='val', download=True)
    Flowers102(root=data_dir, split='test', download=True)
    print(f"Данные успешно скачаны и сохранены в папку '{data_dir}'.")


class FlowerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Трансформации для обучающего набора с аугментациями
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Трансформации для валидационного и тестового наборов
        self.val_test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def setup(self, stage=None):
        """
        Загружает датасеты для обучения, валидации и теста.
        """
        self.train_dataset = Flowers102(
            root=self.data_dir,
            split='train',
            transform=self.train_transforms,
            download=True
        )
        self.val_dataset = Flowers102(
            root=self.data_dir,
            split='val',
            transform=self.val_test_transforms,
            download=True
        )
        self.test_dataset = Flowers102(
            root=self.data_dir,
            split='test',
            transform=self.val_test_transforms,
            download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

