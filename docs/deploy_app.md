# The web app

A Streamlit app ([app/](../app/)) that lets anyone use the models without code:

- **Field monitor:** the 1,254 wheat and rapeseed fields of 2024–25 on a map, as an agronomist would
  have seen them on any day of the season, with a list of fields to check.
- **Try the gap-filling:** hide a period of a real field and let the model API rebuild it.
- **Land-cover map:** the Reims area classified in four seasons of 2025, next to ESA WorldCover, and
  classification of your own tiles.

```
Browser  →  Streamlit app (Streamlit Community Cloud)  →  model API (AWS Lambda)
            reads precomputed results in app/data/        live predictions
```

The app holds no model. The map pages read results that a batch job computed with the same models
the API serves, as a monitoring service would after each satellite pass. The live pages call the API.

## Run it locally

```bash
python -m venv .venv-app && .venv-app/bin/pip install -r app/requirements.txt
.venv-app/bin/streamlit run app/streamlit_app.py      # http://localhost:8501
```

By default it calls the API on AWS Lambda. To use a local one: `API_URL=http://localhost:8000 streamlit
run app/streamlit_app.py`.

## Rebuild its data

`app/data/` and `app/samples/` (about 7 MB) are committed, so the app works without the raw data.
After retraining a model, rebuild them from the research environment:

```bash
python scripts/train_gapfill.py --seasons 2023-24   # the deployed models never saw 2024-25
python scripts/build_app_data.py                     # fields, land-cover maps, sample tiles
python scripts/build_app_data.py --fields-only       # only the field results (a minute)
```

The land-cover part downloads four full Level-1C scenes from the Copernicus Data Space once (about
150 MB each, cached in `data/interim/realworld/`), with the credentials in `.env`.

## Deploy it (Streamlit Community Cloud, free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, and allow it to read your
   repositories.
2. **Create app → Deploy a public app from GitHub**:
   - Repository: `MedChahb/Sentinel-agri-ai`, branch `main`.
   - Main file path: `app/streamlit_app.py`. Streamlit then installs `app/requirements.txt`, the one
     next to the app, not the research `requirements.txt` at the root.
   - App URL: choose a name, for example `sentinel-agri-ai`.
   - **Advanced settings → Python 3.12.**
3. **Deploy.** The first build takes a few minutes. Each push to `main` redeploys the app.

The app needs no secret: the API is public. To point it at another API, add `API_URL = "https://..."`
in the app's **Settings → Secrets**.

After some time without visitors, Streamlit Cloud puts the app to sleep; the next visitor clicks a
button to wake it (about a minute). The app also wakes the Lambda API as soon as someone opens it, so
the first live prediction is usually ready by the time it is needed.
