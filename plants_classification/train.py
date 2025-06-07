import os
import subprocess

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics import Accuracy, F1Score
from torchvision.models import ResNet50_Weights, resnet50

from plants_classification.data import FlowerDataModule


def get_git_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_hash = "unknown"
    return commit_hash


def save_metrics_plots(metrics_history, plots_dir, epoch):
    os.makedirs(plots_dir, exist_ok=True)

    # Проверяем, что есть непустые метрики
    lens = [len(v) for v in metrics_history.values() if len(v) > 0]
    if not lens:
        return []
    length = min(lens)

    epochs = list(range(1, length + 1))

    for key in metrics_history:
        metrics_history[key] = metrics_history[key][:length]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss through Epoch {epoch}")
    plt.legend()
    loss_plot_path = os.path.join(plots_dir, f"loss_epoch_{epoch}.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, metrics_history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy through Epoch {epoch}")
    plt.legend()
    acc_plot_path = os.path.join(plots_dir, f"accuracy_epoch_{epoch}.png")
    plt.savefig(acc_plot_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history["train_f1"], label="Train F1")
    plt.plot(epochs, metrics_history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score through Epoch {epoch}")
    plt.legend()
    f1_plot_path = os.path.join(plots_dir, f"f1_epoch_{epoch}.png")
    plt.savefig(f1_plot_path)
    plt.close()

    return [loss_plot_path, acc_plot_path, f1_plot_path]


class MetricsPlotCallback(Callback):
    def __init__(self, plots_dir):
        super().__init__()
        self.plots_dir = plots_dir

    def on_train_epoch_end(self, trainer, pl_module):
        # Ждём, пока будут метрики валидации
        if not pl_module.val_loss_history:
            return
        if not pl_module.train_loss_history:
            return

        metrics_history = {
            "train_loss": pl_module.train_loss_history,
            "val_loss": pl_module.val_loss_history,
            "train_acc": pl_module.train_acc_history,
            "val_acc": pl_module.val_acc_history,
            "train_f1": pl_module.train_f1_history,
            "val_f1": pl_module.val_f1_history,
        }
        print(metrics_history)

        plot_paths = save_metrics_plots(
            metrics_history, self.plots_dir, trainer.current_epoch + 1
        )

        if mlflow.active_run():
            for path in plot_paths:
                mlflow.log_artifact(path)


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
            torch.nn.Linear(512, num_classes),
        )

        # Отдельные метрики для train и val
        m = "multiclass"
        self.train_accuracy = Accuracy(num_classes=num_classes, task=m)
        self.train_f1 = F1Score(
            num_classes=num_classes, average="macro", task="multiclass"
        )

        self.val_accuracy = Accuracy(num_classes=num_classes, task=m)
        self.val_f1 = F1Score(
            num_classes=num_classes, average="macro", task="multiclass"
        )

        # Накопление метрик по эпохам
        self._train_loss_epoch = []
        self._train_acc_epoch = []
        self._train_f1_epoch = []

        self._val_loss_epoch = []
        self._val_acc_epoch = []
        self._val_f1_epoch = []

        # История метрик по эпохам
        self.train_loss_history = []
        self.train_acc_history = []
        self.train_f1_history = []

        self.val_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.train_accuracy(preds, y)
        f1 = self.train_f1(preds, y)
        t = True
        f = False

        self.log("train_loss", loss, prog_bar=t, on_step=f, on_epoch=t)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        self._train_loss_epoch.append(loss.item())
        self._train_acc_epoch.append(acc.item())
        self._train_f1_epoch.append(f1.item())

        return loss

    def on_train_epoch_end(self, outputs=None):
        if self._train_loss_epoch:
            self.train_loss_history.append(
                sum(self._train_loss_epoch) / len(self._train_loss_epoch)
            )
            self._train_loss_epoch.clear()
        if self._train_acc_epoch:
            self.train_acc_history.append(
                sum(self._train_acc_epoch) / len(self._train_acc_epoch)
            )
            self._train_acc_epoch.clear()
        if self._train_f1_epoch:
            self.train_f1_history.append(
                sum(self._train_f1_epoch) / len(self._train_f1_epoch)
            )
            self._train_f1_epoch.clear()

        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        self._val_loss_epoch.append(loss.item())
        self._val_acc_epoch.append(acc.item())
        self._val_f1_epoch.append(f1.item())

    def on_validation_epoch_end(self, outputs=None):
        if self._val_loss_epoch:
            self.val_loss_history.append(
                sum(self._val_loss_epoch) / len(self._val_loss_epoch)
            )
            self._val_loss_epoch.clear()
        if self._val_acc_epoch:
            self.val_acc_history.append(
                sum(self._val_acc_epoch) / len(self._val_acc_epoch)
            )
            self._val_acc_epoch.clear()
        if self._val_f1_epoch:
            self.val_f1_history.append(
                sum(self._val_f1_epoch) / len(self._val_f1_epoch)
            )
            self._val_f1_epoch.clear()

        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
    )

    os.makedirs(cfg.logging.plots_dir, exist_ok=True)

    mlflow_logger.log_hyperparams(cfg_dict)
    mlflow_logger.experiment.log_param(
        mlflow_logger.run_id, "git_commit", get_git_commit_hash()
    )

    datamodule = FlowerDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        resize=cfg.data.resize,
        crop_size=cfg.data.crop_size,
    )
    datamodule.setup()

    model = FlowerResNet50(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        monitor="val_loss",
        filename="best-{epoch}",
    )

    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    plot_cb = MetricsPlotCallback(plots_dir=cfg.logging.plots_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, early_stop_cb, plot_cb],
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="final_model",
        registered_model_name="FlowerClassifier",
    )


if __name__ == "__main__":
    main()
