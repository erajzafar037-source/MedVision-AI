# Development Guide

## Environment

Use Python 3.11 or 3.12 and create an isolated virtual environment.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

The project expects the NIH ChestX-ray14 source files under `data/`. The dataset is not included in this repository.

## Prepare data and train

```bash
python prepare_data.py
python train.py
python evaluate.py
```

The preprocessing step creates `data/train.csv`, `data/val.csv`, and `data/test.csv` using patient-level splits. Evaluation writes `models/test_metrics.json` and `models/roc_curve.png`.

## Run the application

```bash
streamlit run app.py
```

The app reads test metrics from `models/test_metrics.json`. If evaluation has not been run, performance fields are shown as “Not generated” instead of displaying manually entered scores.

## Tests

The unit tests use temporary directories and synthetic images; they do not require the NIH dataset.

```bash
python -m pytest -q
python -m compileall -q app.py dataset.py evaluate.py metrics.py prepare_data.py train.py
```

## Data and model hygiene

- Keep the original dataset under `data/`; it is excluded from version control.
- Do not commit identifiable patient images or protected health information.
- The generated metrics file is a local evaluation artefact and should be regenerated for the checkpoint and split being reported.
- Do not claim clinical validity or use the demo output for care decisions.
