import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from dataset import ChestXrayDataset
from metrics import calculate_binary_metrics


BATCH_SIZE = 32
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/best_model.pt")
METRICS_PATH = Path("models/test_metrics.json")
ROC_CURVE_PATH = Path("models/roc_curve.png")


def main():
    print("Using device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Run train.py first."
        )

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = ChestXrayDataset(
        csv_file="data/test.csv",
        data_folder="data",
        transform=image_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.sigmoid(logits).squeeze(1)
            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    results = calculate_binary_metrics(
        all_labels,
        all_probabilities,
        threshold=THRESHOLD,
    )

    print("\nTest results")
    for metric_name in (
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "accuracy",
    ):
        print(f"{metric_name.replace('_', ' ').title():18s}: {results[metric_name]:.4f}")
    print(f"Samples evaluated: {results['sample_count']}")
    print("Class counts:", results["class_counts"])
    print("\nConfusion matrix")
    print(np.asarray(results["confusion_matrix"]))
    print("\nClassification report")
    print(
        classification_report(
            all_labels,
            [int(score >= THRESHOLD) for score in all_probabilities],
            labels=[0, 1],
            target_names=["No Finding", "Pneumonia"],
            digits=4,
            zero_division=0,
        )
    )

    # This report is generated from the test run; it is not a hand-entered UI constant.
    report = {
        "task": (
            "NIH ChestX-ray14 subset: No Finding versus records whose "
            "Finding Labels contain Pneumonia"
        ),
        "checkpoint": str(MODEL_PATH),
        "threshold": THRESHOLD,
        "metrics": results,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved test metrics to {METRICS_PATH}")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        all_labels,
        all_probabilities,
    )
    plt.figure(figsize=(7, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {results['roc_auc']:.3f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Chest X-Ray Model ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=200)
    plt.close()
    print(f"Saved ROC curve to {ROC_CURVE_PATH}")


if __name__ == "__main__":
    main()
