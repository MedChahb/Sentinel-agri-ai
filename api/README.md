# The model API

A FastAPI service that serves both models. It runs on **AWS Lambda**:
[interactive documentation](https://cga6hukfpyansgulvry42cg3xi0ygxaa.lambda-url.eu-north-1.on.aws/docs).

| Endpoint | Input | Output |
|---|---|---|
| `POST /classify` | A 13-band, 64 × 64 Sentinel-2 Level-1C GeoTIFF, EuroSAT band order | The land-cover class and the probabilities of the 10 classes. Level-2A tiles are refused (422), unless `?allow_level2a=true`: then they are classified with a warning. |
| `POST /gapfill` | Crop (`Winter wheat` or `Winter rapeseed`), mode (`after_season` or `real_time`), the clear NDVI values of one field, optionally its Sentinel-1 VV and VH | NDVI every 5 days from 1 October to 31 August, with a range calibrated to hold the truth about 80 % of the time (`ndvi_low`, `ndvi_high`). The radar models are used from 10 radar passes. |
| `GET /health` | | Which models are loaded |

```bash
URL=https://cga6hukfpyansgulvry42cg3xi0ygxaa.lambda-url.eu-north-1.on.aws
curl -F "file=@app/samples/eurosat_test_industrial.tif" $URL/classify
curl -H "Content-Type: application/json" $URL/gapfill -d '{"crop": "Winter wheat", "mode": "real_time",
  "ndvi": [{"date": "2024-10-20", "ndvi": 0.21}, {"date": "2025-03-14", "ndvi": 0.52}]}'
```

The first request after a quiet period takes a few seconds: Lambda starts the container and the API
loads its models. The next ones take a fraction of a second.

## How it is built

- **No PyTorch in production.** The classifier is exported to ONNX (`scripts/export_classifier.py`,
  checked against PyTorch on 500 tiles) and the gap-filling models are XGBoost files. The image is
  275 MB compressed, against 1.9 GB with PyTorch.
- **Models trained on 2023–24 only** (`scripts/train_gapfill.py --seasons 2023-24`), so that the web
  app's 2024–25 demo season is one they have never seen. In production, train on every past season.
- **One image, three places:** the same Docker image runs on a laptop, on AWS Lambda (through the AWS
  Lambda Web Adapter) and on ECS.

## Run it locally

```bash
python scripts/export_classifier.py               # models/eurosat_resnet18.onnx (+ .json)
python scripts/train_gapfill.py --seasons 2023-24 # models/gapfill/
docker compose up -d --build api                  # -> http://localhost:8000/docs
```

Without the models, the API still starts: `/health` shows which are missing, and their endpoint answers
503. Tests (`pytest -q`) use small stand-in models, so they run anywhere, as in CI.

## Deploy it

Step by step, with costs and clean-up: [docs/deploy_aws.md](../docs/deploy_aws.md). AWS Lambda gives a
public URL within its free allowance; ECS Express Mode is the always-on alternative, at about $2 a day.
