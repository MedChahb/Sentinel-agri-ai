"""Export the part 1 classifier to ONNX and check it against PyTorch on EuroSAT test tiles.

Usage: python scripts/export_classifier.py
"""

import numpy as np
import onnxruntime as ort
import torch

from sentinel_agri.data import eurosat
from sentinel_agri.export import CLASSIFIER_ONNX, export_classifier, softmax


def main(n_tiles: int = 500) -> None:
    model = export_classifier()
    session = ort.InferenceSession(str(CLASSIFIER_ONNX), providers=["CPUExecutionProvider"])
    test = eurosat("test")  # file order, as the API receives tiles
    rng = np.random.default_rng(0)
    tiles = np.stack(
        [test[i]["image"].numpy() for i in rng.choice(len(test), n_tiles, replace=False)]
    )
    with torch.inference_mode():
        reference = softmax(model(torch.from_numpy(tiles)).numpy())
    exported = softmax(session.run(None, {"tiles": tiles.astype(np.float32)})[0])
    same_class = (reference.argmax(1) == exported.argmax(1)).mean()
    print(f"Exported {CLASSIFIER_ONNX} ({CLASSIFIER_ONNX.stat().st_size / 1e6:.0f} MB)")
    gap = np.abs(reference - exported).max()
    print(
        f"{n_tiles} test tiles: same class {same_class:.1%}, max probability difference {gap:.2e}"
    )


if __name__ == "__main__":
    main()
