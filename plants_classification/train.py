import os
import subprocess
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from torchmetrics import Accuracy, F1Score

import hydra
from omegaconf import DictConfig

from plants_classification.data import FlowerDataModule

import mlflow.pytorch


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        commit_hash = "unknown"
    return commit_hash


def save_metrics_plots(metrics_history, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    epochs = list(range(1, len(metrics_history['train_loss']) + 1))

    # График потерь
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics_history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    loss_plot_path = os.path.join(plots_dir, 'loss.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # График точности
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics_history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    acc_plot_path = os.path.join(plots_dir, 'accuracy.png')
    plt.savefig(acc_plot_path)
    plt.close()

    # График F1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history['train_f1'], label='Train F1')
    plt.plot(epochs, metrics_history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over epochs')
    plt.legend()
    f1_plot_path = os.path.join(plots_dir, 'f1_score.png')
    plt.savefig(f1_plot_path)
    plt.close()

    return [loss_plot_path, acc_plot_path, f1_plot_path]


class FlowerResNet50(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, freeze_backbone):
        super().__init__()
        self.save_hyperparameters()

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.fc.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )

        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.f1 = F1Score(num_classes=num_classes, average='macro', task="multiclass")

        # Для хранения метрик по эпохам
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_f1_history = []
        self.val_f1_history = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)

        # Сохраняем для графиков
        self.train_loss_history.append(loss.item())
        self.train_acc_history.append(acc.item())
        self.train_f1_history.append(f1.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        # Сохраняем для графиков
        self.val_loss_history.append(loss.item())
        self.val_acc_history.append(acc.item())
        self.val_f1_history.append(f1.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


@hydra.main(version_base = None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Инициализация MLflowLogger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri
    )

    # Логирование git commit id и гиперпараметров
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "git_commit", get_git_commit_hash())
    # Логируем все параметры из конфига рекурсивно
    def log_params_recursive(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                log_params_recursive(f"{prefix}.{k}" if prefix else k, v)
            else:
                mlflow_logger.experiment.log_param(mlflow_logger.run_id, f"{prefix}.{k}" if prefix else k, str(v))
    log_params_recursive("", dict(cfg))

    # Датамодуль
    datamodule = FlowerDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        resize=cfg.data.resize,
        crop_size=cfg.data.crop_size
    )
    datamodule.setup()

    # Модель
    model = FlowerResNet50(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        freeze_backbone=cfg.model.freeze_backbone
    )

    # Колбеки
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint"
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5)

    # Тренер
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=mlflow_logger,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices
    )

    trainer.fit(model, datamodule=datamodule)

    # После тренировки сохраняем графики и логируем их в MLflow
    metrics_history = {
        "train_loss": model.train_loss_history,
        "val_loss": model.val_loss_history,
        "train_acc": model.train_acc_history,
        "val_acc": model.val_acc_history,
        "train_f1": model.train_f1_history,
        "val_f1": model.val_f1_history,
    }
    plot_paths = save_metrics_plots(metrics_history, cfg.logging.plots_dir)

    for plot_path in plot_paths:
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, plot_path)

    # Логируем модель
    mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()

