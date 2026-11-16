from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """Dataset adapter for the prepared NIH ChestX-ray14 binary subset."""

    def __init__(self, csv_file: str, data_folder: str, transform=None):
        self.data_folder = Path(data_folder)
        self.transform = transform

        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV file: {csv_path}")
        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Could not find data folder: {self.data_folder}"
            )

        self.labels = pd.read_csv(csv_path)
        required_columns = {"Image Index", "Finding Labels", "label"}
        missing_columns = required_columns - set(self.labels.columns)
        if missing_columns:
            raise KeyError(
                f"CSV is missing required columns: {sorted(missing_columns)}. "
                "Run prepare_data.py again."
            )

        # A filename-only index is used because the source CSV stores filenames.
        # Detect collisions rather than silently selecting the last matching file.
        image_files = sorted(
            path
            for path in self.data_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".png"
        )
        self.image_paths = {}
        duplicate_names = set()
        for image_path in image_files:
            if image_path.name in self.image_paths:
                duplicate_names.add(image_path.name)
            else:
                self.image_paths[image_path.name] = image_path

        if duplicate_names:
            examples = ", ".join(sorted(duplicate_names)[:5])
            raise ValueError(
                "Duplicate image filenames found in the data folder. "
                "The dataset CSV cannot distinguish these paths. "
                f"Examples: {examples}"
            )

        if not self.image_paths:
            raise RuntimeError(
                f"No PNG images were found inside {self.data_folder}."
            )

        self.labels = self.labels[
            self.labels["Image Index"].isin(self.image_paths)
        ].copy().reset_index(drop=True)

        if self.labels.empty:
            raise RuntimeError(
                "No CSV rows matched the image files. "
                "Check that the X-ray folders are inside the data folder."
            )

        numeric_labels = pd.to_numeric(self.labels["label"], errors="coerce")
        if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
            raise ValueError("Dataset labels must contain only binary values 0 or 1.")
        self.labels["label"] = numeric_labels.astype(int)

        print(f"Loaded {len(self.labels)} labeled X-rays.")
        label_counts = self.labels["label"].value_counts().sort_index()
        print("Class counts:")
        print(f"No Finding: {label_counts.get(0, 0)}")
        print(f"Pneumonia:  {label_counts.get(1, 0)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.labels):
            raise IndexError(f"Index {index} is outside dataset range.")

        row = self.labels.iloc[index]
        image_name = row["Image Index"]
        image_path = self.image_paths[image_name]

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not open image: {image_path}"
            ) from error

        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)

        return image, label
