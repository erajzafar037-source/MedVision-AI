# MedVision AI — Model Card

## Summary

MedVision AI is an educational chest X-ray computer-vision research prototype. The repository contains a ResNet-18 binary classifier, a Streamlit interface, and a Grad-CAM visualization. It is not a medical device and must not be used for diagnosis, screening, triage, or treatment decisions.

## Intended task

The current task is narrowly defined as:

- **Positive class:** records whose NIH ChestX-ray14 `Finding Labels` field contains `Pneumonia`.
- **Negative class:** records whose `Finding Labels` field is exactly `No Finding`.
- **Excluded records:** records with other findings that do not meet either rule.

This is not a general-purpose pneumonia detector and is not a multi-label model.

## Data

The project expects the NIH ChestX-ray14 image files and `Data_Entry_2017.csv` under `data/`. The dataset is not distributed in this repository. The preprocessing script splits at patient level to reduce leakage of images from the same patient across train, validation, and test sets.

The source labels are dataset annotations, not independently adjudicated clinical ground truth. Dataset composition and label quality limit what can be inferred from model results.

## Model and inputs

- Architecture: ResNet-18 with a single-logit output layer.
- Framework: PyTorch / Torchvision.
- Input: PNG or JPEG image converted to RGB and resized to 224 × 224.
- Output: sigmoid score for the positive class, thresholded in the demo UI.

The score is an uncalibrated model output, not a patient-specific clinical probability.

## Evaluation

Run `python evaluate.py` after preparing the dataset and training or obtaining the checkpoint. The script writes `models/test_metrics.json` and `models/roc_curve.png`. The Streamlit application reads the generated metrics file; it no longer assumes hard-coded test metrics exist.

Metrics depend on the exact checkpoint, prepared split, and threshold. Report the sample count, class counts, threshold, and confusion matrix alongside summary metrics. Do not tune a threshold on the held-out test set.

## Explainability limitations

Grad-CAM highlights image regions associated with model activations for a prediction. It does not identify a lesion, prove the model used medically valid evidence, or establish causal reasoning. Heatmaps can highlight acquisition artefacts, labels, borders, devices, or other confounders.

## Risks and limitations

- Class imbalance can make precision and other threshold-dependent metrics difficult to interpret.
- A patient-level random split is not external validation.
- No prospective, multi-site, subgroup, calibration, or clinical validation is claimed.
- Performance can shift across hospitals, scanners, populations, and acquisition protocols.
- The app is intended for controlled research demonstrations using non-identifiable sample images.

## Privacy

Uploaded images are processed in the Streamlit session and are not intentionally written to application-managed storage. Do not upload identifiable patient images or protected health information to a public demo. A production deployment would require a documented privacy threat model, access controls, retention policy, transport security, and jurisdiction-specific review.

## Attribution and licensing

The repository retains attribution to the original project author, Leya Glazer. This codebase is derived from that original project; review the upstream repository and its licensing terms before redistribution or reuse. No license file is present in this repository at the time of this update.
