"""Metrics and figures shared by all part 1 models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from sentinel_agri.config import EUROSAT_CLASSES


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """True labels and predicted class probabilities over a whole loader."""
    model.eval()
    device = next(model.parameters()).device
    labels, probs = [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(labels), np.concatenate(probs)


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, macro F1 and the F1 of each class."""
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(EUROSAT_CLASSES)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        **{f"f1_{c}": f for c, f in zip(EUROSAT_CLASSES, per_class, strict=True)},
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "", ax=None):
    """Confusion matrix normalised by row: each row shows where the tiles of a true class went."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EUROSAT_CLASSES)), normalize="true")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm * 100, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] >= 0.005:
                ax.text(
                    j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.6 else "black",
                )  # fmt: skip
    ax.set_xticks(range(len(EUROSAT_CLASSES)), EUROSAT_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(EUROSAT_CLASSES)), EUROSAT_CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label="% of the true class")
    return ax


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, n: int = 8) -> pd.DataFrame:
    """The most frequent (true class -> predicted class) errors."""
    errors = pd.DataFrame({"true": y_true, "predicted": y_pred})
    errors = errors[errors.true != errors.predicted]
    counts = errors.value_counts().head(n).reset_index(name="count")
    counts["true"] = counts["true"].map(dict(enumerate(EUROSAT_CLASSES)))
    counts["predicted"] = counts["predicted"].map(dict(enumerate(EUROSAT_CLASSES)))
    return counts
