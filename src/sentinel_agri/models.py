"""Part 1 deep models and the Lightning module that trains them."""

import time

import kornia.augmentation as K
import lightning as L
import timm
import torch
from torch import nn
from torchgeo.models import ResNet18_Weights, resnet18

from sentinel_agri.config import EUROSAT_CLASSES

N_CLASSES = len(EUROSAT_CLASSES)


class SmallCNN(nn.Module):
    """Four conv blocks (64 -> 32 -> 16 -> 8 -> 4 pixels), global average pooling, linear head.

    About 1.2 M parameters, against 11 M for ResNet-18.
    """

    def __init__(self, in_channels: int, num_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def resnet18_sentinel2() -> nn.Module:
    """ResNet-18 pretrained with MoCo on SSL4EO-S12 (13-band Sentinel-2 L1C), new 10-class head.

    Expects the 13 bands in wavelength order (B8A after B08) and values divided by 10,000.
    """
    return resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO, num_classes=N_CLASSES)


def resnet18_imagenet() -> nn.Module:
    """ResNet-18 pretrained on ImageNet (3-channel RGB photos), new 10-class head."""
    return timm.create_model("resnet18", pretrained=True, num_classes=N_CLASSES)


class EpochSummary(L.Callback):
    """Prints one line per epoch, instead of Lightning's progress bars.

    A star marks the epochs where the validation accuracy is the best so far, which are the
    epochs ModelCheckpoint saves.
    """

    def on_train_start(self, trainer, pl_module) -> None:
        self.best_val_acc = 0.0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        is_best = m["val_acc"] > self.best_val_acc
        self.best_val_acc = max(self.best_val_acc, m["val_acc"])
        print(
            f"epoch {trainer.current_epoch + 1:>2}/{trainer.max_epochs}"
            f" | train loss {m['train_loss']:.3f} acc {m['train_acc']:.3f}"
            f" | val loss {m['val_loss']:.3f} acc {m['val_acc']:.3f}"
            f" | {time.perf_counter() - self.start:4.0f} s{' *' if is_best else ''}"
        )


class LitClassifier(L.LightningModule):
    """Trains any image classifier on raw EuroSAT values.

    The normalisation `(x - mean) / std` is part of the module, so the exported model always
    carries the preprocessing it was trained with.
    """

    def __init__(
        self,
        net: nn.Module,
        mean: list[float],
        std: list[float],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        # A satellite tile has no "up": flips and 90-degree rotations give equally valid tiles.
        self.augment = (
            nn.Sequential(
                K.RandomHorizontalFlip(), K.RandomVerticalFlip(), K.RandomRotation90(times=(0, 3))
            )
            if augment
            else nn.Identity()
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        if stage == "train":
            x = self.augment(x)
        logits = self(x)
        loss = self.loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
