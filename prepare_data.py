from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FOLDER = Path("data")
SOURCE_CSV = DATA_FOLDER / "Data_Entry_2017.csv"
RANDOM_STATE = 42


def create_patient_splits(
    dataframe: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Split rows by patient so images from one patient cannot cross splits."""
    required_columns = {"Patient ID", "Image Index", "label"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(f"Missing columns for patient split: {sorted(missing_columns)}")
    if dataframe["Patient ID"].isna().any():
        raise ValueError("Patient ID contains missing values; cannot guarantee patient-level separation.")

    patient_ids = dataframe["Patient ID"].drop_duplicates().to_numpy()
    if len(patient_ids) < 3:
        raise ValueError("At least three distinct patients are required for train/validation/test splits.")

    train_patients, remaining_patients = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=random_state,
    )
    val_patients, test_patients = train_test_split(
        remaining_patients,
        test_size=0.50,
        random_state=random_state,
    )

    splits = {
        "train": dataframe[dataframe["Patient ID"].isin(train_patients)].copy(),
        "val": dataframe[dataframe["Patient ID"].isin(val_patients)].copy(),
        "test": dataframe[dataframe["Patient ID"].isin(test_patients)].copy(),
    }

    if any(split.empty for split in splits.values()):
        raise ValueError("A patient-level split is empty; add more patient records.")

    patient_sets = {
        name: set(split["Patient ID"].unique())
        for name, split in splits.items()
    }
    if (
        patient_sets["train"] & patient_sets["val"]
        or patient_sets["train"] & patient_sets["test"]
        or patient_sets["val"] & patient_sets["test"]
    ):
        raise RuntimeError("Patient leakage detected between dataset splits.")

    return splits


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Could not find: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)
    required_columns = {"Finding Labels", "Patient ID", "Image Index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    if df["Patient ID"].isna().any():
        raise ValueError("Source metadata contains missing Patient ID values.")

    # Positive: Pneumonia appears in the multi-label finding string.
    pneumonia_mask = df["Finding Labels"].str.contains(
        "Pneumonia", regex=False, na=False
    )
    # Negative: explicitly labeled No Finding. Other findings are excluded.
    no_finding_mask = df["Finding Labels"].eq("No Finding")
    filtered = df[pneumonia_mask | no_finding_mask].copy()
    filtered["label"] = pneumonia_mask.loc[filtered.index].astype(int)

    if filtered.empty or filtered["label"].nunique() != 2:
        raise ValueError(
            "The selected records must include both No Finding and Pneumonia classes."
        )

    splits = create_patient_splits(filtered)
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        output_path = DATA_FOLDER / f"{name}.csv"
        split_df.to_csv(output_path, index=False)
        counts = split_df["label"].value_counts().sort_index()
        print(f"\n{name.title()} images: {len(split_df)}")
        print(
            counts.rename(index={0: "No Finding", 1: "Pneumonia"})
        )
        print(f"Saved {output_path}")

    print("\nPatient-level split check passed: no patient overlap found.")


if __name__ == "__main__":
    main()
