"""API tests, with small stand-in models: they need neither the trained models nor PyTorch."""

import importlib
import json
from datetime import date, timedelta

import numpy as np
import onnx
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from sentinel_agri.gapfill import SERVED_FEATURES

CLASSES = [f"class_{i}" for i in range(10)]
CROPS = ["Winter wheat", "Winter rapeseed"]


# --- Stand-in models ----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def classifier_path(tmp_path_factory):
    """ONNX graph with the real input/output names: band means -> linear layer -> 10 logits."""
    path = tmp_path_factory.mktemp("models") / "classifier.onnx"
    weights = np.random.default_rng(0).normal(size=(13, 10)).astype(np.float32) / 1000
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["tiles"], ["means"], axes=[2, 3], keepdims=0),
            helper.make_node("MatMul", ["means", "weights"], ["logits"]),
        ],
        "stand_in",
        [helper.make_tensor_value_info("tiles", TensorProto.FLOAT, ["batch", 13, 64, 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(model, path)  # IR 8: readable by every recent onnxruntime
    path.with_suffix(".json").write_text(json.dumps({"classes": CLASSES}))
    return path


@pytest.fixture(scope="session")
def gapfill_dir(tmp_path_factory):
    """Eight small XGBoost models with the real feature names, trained on random data."""
    directory = tmp_path_factory.mktemp("gapfill")
    rng = np.random.default_rng(0)
    metadata = {"trained_on": ["test"], "models": {}}
    for crop in CROPS:
        for (mode, radar), names in SERVED_FEATURES.items():
            names = [f for f in names if f != "rapeseed"]
            x = rng.normal(size=(200, len(names)))
            data = xgb.DMatrix(x, label=rng.uniform(0.1, 0.9, 200), feature_names=names)
            model = xgb.train({"max_depth": 3}, data, num_boost_round=5)
            name = f"{mode}_{crop}_{radar}".replace(" ", "_")
            model.save_model(directory / f"{name}.json")
            metadata["models"][name] = {
                "crop": crop,
                "mode": mode,
                "radar": radar,
                "features": names,
            }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


def make_client(monkeypatch, classifier, gapfill) -> TestClient:
    monkeypatch.setenv("MODEL_PATH", str(classifier))
    monkeypatch.setenv("GAPFILL_DIR", str(gapfill))
    import api.main

    importlib.reload(api.main)  # re-read the paths
    return TestClient(api.main.app)


@pytest.fixture
def client(monkeypatch, classifier_path, gapfill_dir):
    with make_client(monkeypatch, classifier_path, gapfill_dir) as c:
        yield c


@pytest.fixture
def empty_client(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path / "missing.onnx", tmp_path / "missing") as c:
        yield c


# --- Health -------------------------------------------------------------------------------------


def test_health_with_models(client):
    expected = {"status": "ok", "model_loaded": True, "gapfill_loaded": True}
    assert client.get("/health").json() == expected


def test_health_without_models(empty_client):
    expected = {"status": "ok", "model_loaded": False, "gapfill_loaded": False}
    assert empty_client.get("/health").json() == expected


# --- /classify ----------------------------------------------------------------------------------


def make_tiff(bands: int = 13, size: int = 64, data: np.ndarray | None = None) -> bytes:
    if data is None:
        data = np.random.default_rng(0).integers(0, 4000, (bands, size, size), dtype=np.uint16)
    bands, size = data.shape[0], data.shape[1]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", count=bands, height=size, width=size, dtype="uint16",
            transform=from_origin(0, size * 10, 10, 10),
        ) as dst:  # fmt: skip
            dst.write(data)
        return memfile.read()


def classify(client, payload: bytes):
    return client.post("/classify", files={"file": ("tile.tif", payload)})


def test_classify_returns_a_valid_prediction(client):
    response = classify(client, make_tiff())
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASSES
    assert 0.0 <= body["probability"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_classify_without_model_returns_503(empty_client):
    assert classify(empty_client, make_tiff()).status_code == 503


@pytest.mark.parametrize(
    "payload", [make_tiff(bands=3), make_tiff(size=32), b"not a tiff"], ids=["rgb", "small", "junk"]
)
def test_classify_rejects_bad_input(client, payload):
    assert classify(client, payload).status_code == 422


def level_2a_like(b10_constant: bool, b01_factor: float) -> bytes:
    """A 13-band tile (file order) with Level-2A traits: constant B10 and/or B01 darker than B02."""
    data = np.random.default_rng(1).integers(500, 3000, (13, 64, 64)).astype(np.uint16)
    data[0] = (data[1] * b01_factor).astype(np.uint16)  # B01 relative to B02
    if b10_constant:
        data[9] = 12  # B10 filled with a constant, as a Level-2A sender must
    return make_tiff(data=data)


@pytest.mark.parametrize(
    "b10_constant, b01_factor", [(True, 1.2), (False, 0.5)], ids=["constant-b10", "dark-b01"]
)
def test_classify_rejects_level_2a(client, b10_constant, b01_factor):
    response = classify(client, level_2a_like(b10_constant, b01_factor))
    assert response.status_code == 422
    assert "Level-2A" in response.json()["detail"]


def test_classify_accepts_level_1c_like_tile(client):
    assert classify(client, level_2a_like(False, 1.2)).status_code == 200


# --- /gapfill -----------------------------------------------------------------------------------


def season_request(mode="after_season", n_radar=40, crop="Winter wheat"):
    start = date(2024, 10, 1)
    clear_days = [3, 9, 40, 130, 150, 160, 175, 200, 215, 230, 280, 300]
    ndvi = [
        {"date": str(start + timedelta(days=d)), "ndvi": 0.2 + 0.6 * np.sin(d / 110)}
        for d in clear_days
    ]
    radar = [
        {
            "date": str(start + timedelta(days=2 * k)),
            "vv_db": -10.0,
            "vh_db": -16.0,
            "orbit": 37 + 22 * (k % 2),
        }
        for k in range(n_radar)
    ]
    return {"crop": crop, "mode": mode, "ndvi": ndvi, "radar": radar}


def first_and_last_observed(body):
    observed = body["observed"]
    return observed.index(True), len(observed) - 1 - observed[::-1].index(True)


def test_gapfill_after_season_fills_between_observations(client):
    response = client.post("/gapfill", json=season_request())
    assert response.status_code == 200
    body = response.json()
    assert body["dates"][0] == "2024-10-01" and len(body["dates"]) == 67  # 5-day steps to 31 Aug
    first, last = first_and_last_observed(body)
    assert all(v is not None for v in body["ndvi"][first : last + 1])  # every gap in between
    assert all(v is None for v in body["ndvi"][last + 1 :])  # nothing after the last observation
    assert body["radar_used"] is True


def test_gapfill_real_time_extends_after_the_last_observation(client):
    body = client.post("/gapfill", json=season_request(mode="real_time")).json()
    _, last = first_and_last_observed(body)
    assert all(v is not None for v in body["ndvi"][last:])


def test_gapfill_uses_the_no_radar_model_without_enough_radar(client):
    body = client.post("/gapfill", json=season_request(n_radar=3, crop="Winter rapeseed")).json()
    assert body["radar_used"] is False


def test_gapfill_rejects_dates_outside_one_season(client):
    request = season_request()
    request["ndvi"].append({"date": "2026-01-01", "ndvi": 0.5})
    assert client.post("/gapfill", json=request).status_code == 422


def test_gapfill_rejects_an_unknown_crop(client):
    assert client.post("/gapfill", json=season_request(crop="Maize")).status_code == 422


def test_gapfill_without_models_returns_503(empty_client):
    assert empty_client.post("/gapfill", json=season_request()).status_code == 503
