import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics import Accuracy, F1Score
import hydra
from omegaconf import DictConfig
from plants_classification.data import FlowerDataModule


class FlowerResNet50(pl.LightningModule):
    def __init__(self, num_classes=102, learning_rate=1e-3, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()

        # Загружаем предобученную ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=weights)

        # Замораживаем начальные слои, если требуется
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Заменяем последний fully connected слой под нашу задачу
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # Метрики
        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.f1 = F1Score(num_classes=num_classes, average='macro', task="multiclass")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", acc, prog_bar=True)
        self.log("train/f1_score", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/f1_score", f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Можно добавить scheduler, если нужно
        return optimizer


@hydra.main(version_base = "1.1",config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Инициализация DataModule с параметрами из конфига
    data_module = FlowerDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    data_module.setup()

    # Инициализация модели с параметрами из конфига
    model = FlowerResNet50(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        freeze_backbone=cfg.model.freeze_backbone
    )

    # Логгер MLflow
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri
    )

    # Колбэки: ранняя остановка и сохранение лучшей модели
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=5,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=cfg.training.checkpoint_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=mlflow_logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()

