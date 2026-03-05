# Animal Classifier with Keras

A small ML project that combines:
- model training and evaluation in notebooks
- standalone batch image testing in a separate notebook
- a Flask web app for model stats, charts, sample predictions, and JPEG upload inference

## What this project includes

- `main_new.ipynb`: end-to-end training/evaluation workflow (Animals-10 style dataset)
- `testing.ipynb`: inference-only notebook for testing folders of images (kept separate from training)
- `app.py`: Flask app serving pages and JSON APIs
- `templates/` + `static/`: frontend for Home/About pages, charts, and prediction UI
- `func.py`: reusable data utilities and preprocessing helpers
- `models/`: saved Keras models (`model.keras`, `model_fixed.keras`)
- `histories/model-history.npy`: saved training history used for chart rendering

## Project structure

```text
.
|-- app.py
|-- func.py
|-- main_new.ipynb
|-- testing.ipynb
|-- requirements.txt
|-- models/
|   |-- model.keras
|   `-- model_fixed.keras
|-- histories/
|   `-- model-history.npy
|-- templates/
|   |-- base.html
|   |-- index.html
|   `-- about.html
`-- static/
    |-- images/
    |-- scripts/
    `-- styles/
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the web app

```bash
python app.py
```

Then open:
- `http://127.0.0.1:5000/` (Home)
- `http://127.0.0.1:5000/about` (About)

## Notebook workflows

### 1) Training and charts (`main_new.ipynb`)

- Trains/evaluates a MobileNetV2-based classifier.
- Saves model and history artifacts:
  - `models/model.keras`
  - `histories/model-history.npy`
- If retraining is skipped, notebook loads saved artifacts and renders charts from `model_history`.

### 2) Separate image testing (`testing.ipynb`)

- Loads test images from `IMG_FOLDER`.
- Applies the same inference preprocessing as training:
  - resize to `(224, 224)`
  - RGB
  - scale by `1./255`
- Runs `model.predict(...)`.
- Displays a prediction grid (image + predicted class/confidence).

## API endpoints

- `GET /api/model-stats`
  - Returns summary metrics from saved history + model file metadata.
- `GET /api/model-history`
  - Returns full numeric history series.
- `GET /api/sample-results`
  - Runs predictions on sample images under static sample directories.
- `POST /api/predict-image`
  - Multipart upload (`image`) for live JPEG inference.

## Model compatibility note (important)

If your environment cannot deserialize `models/model.keras` directly (Keras version mismatch), this project already includes a conversion path in notebooks:
- rebuild architecture
- load `model.weights.h5` from legacy archive
- save `models/model_fixed.keras`

If the Flask app fails to load `model.keras`, either:
- replace `models/model.keras` with a compatible saved model, or
- update `MODEL_PATH` in `app.py` to point at `models/model_fixed.keras`.

## Run tests

```bash
python -m unittest test_app.py test_func.py
```

## Notes

- Upload API currently accepts `.jpg` / `.jpeg` only.
- Frontend expects history/model artifacts to exist for full chart/stat rendering.
