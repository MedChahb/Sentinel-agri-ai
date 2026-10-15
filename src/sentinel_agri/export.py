"""Part 3: export the trained models into the light formats the API serves.

- The land-cover classifier (PyTorch) becomes an ONNX file, run by onnxruntime: the API image no
  longer needs PyTorch. Band reordering and normalisation are built into the ONNX graph, so the API
  passes raw tiles in EuroSAT file order.
- The gap-filling models are XGBoost models, saved as JSON by `scripts/train_gapfill.py`.
"""

import json
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn

from sentinel_agri.config import EUROSAT_BANDS, MODELS_DIR

CLASSIFIER_BUNDLE = MODELS_DIR / "eurosat_resnet18.pt"
CLASSIFIER_ONNX = MODELS_DIR / "eurosat_resnet18.onnx"


class RawTileClassifier(nn.Module):
    """Raw tile in file order -> model band order -> normalisation -> network -> class scores."""

    def __init__(self, bundle: dict):
        super().__init__()
        self.net = timm.create_model(
            bundle["arch"], in_chans=len(bundle["bands"]), num_classes=len(bundle["classes"])
        )
        self.net.load_state_dict(bundle["state_dict"])
        order = [EUROSAT_BANDS.index(b) for b in bundle["bands"]]
        self.register_buffer("order", torch.tensor(order))
        self.register_buffer("mean", torch.tensor(bundle["mean"]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(bundle["std"]).view(1, -1, 1, 1))

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(tiles, 1, self.order)
        return self.net((x - self.mean) / self.std)


def export_classifier(bundle_path: Path = CLASSIFIER_BUNDLE, onnx_path: Path = CLASSIFIER_ONNX):
    """Write the ONNX model and, next to it, a JSON file with the class names and input bands."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    model = RawTileClassifier(bundle).eval()
    example = torch.zeros(1, len(EUROSAT_BANDS), 64, 64)
    torch.onnx.export(
        model,
        (example,),
        onnx_path,
        input_names=["tiles"],
        output_names=["logits"],
        dynamic_axes={"tiles": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    metadata = {"classes": bundle["classes"], "input_bands": EUROSAT_BANDS, "tile_size": 64}
    onnx_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return model


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
