from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from dataset import ChestXrayDataset
from prepare_data import create_patient_splits


def write_metadata(path: Path, records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def make_record(image_name: str, label: int) -> dict:
    finding = "Pneumonia" if label == 1 else "No Finding"
    return {
        "Image Index": image_name,
        "Finding Labels": finding,
        "label": label,
    }


def test_dataset_loads_rgb_image_and_binary_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (24, 16), color=120).save(image_dir / "sample.png")
    metadata_path = tmp_path / "labels.csv"
    write_metadata(metadata_path, [make_record("sample.png", 1)])

    dataset = ChestXrayDataset(metadata_path, image_dir)
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.mode == "RGB"
    assert image.size == (24, 16)
    assert label == 1


def test_dataset_ignores_rows_without_matching_files(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "present.png")
    metadata_path = tmp_path / "labels.csv"
    write_metadata(
        metadata_path,
        [
            make_record("present.png", 0),
            make_record("missing.png", 1),
        ],
    )

    dataset = ChestXrayDataset(metadata_path, image_dir)

    assert len(dataset) == 1
    assert dataset.labels.iloc[0]["Image Index"] == "present.png"


def test_dataset_rejects_duplicate_filenames_in_nested_directories(tmp_path):
    image_dir = tmp_path / "images"
    (image_dir / "part_a").mkdir(parents=True)
    (image_dir / "part_b").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_dir / "part_a" / "same.png")
    Image.new("RGB", (8, 8)).save(image_dir / "part_b" / "same.png")
    metadata_path = tmp_path / "labels.csv"
    write_metadata(metadata_path, [make_record("same.png", 0)])

    with pytest.raises(ValueError, match="Duplicate image filenames"):
        ChestXrayDataset(metadata_path, image_dir)


def test_dataset_rejects_non_binary_labels(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "sample.png")
    metadata_path = tmp_path / "labels.csv"
    invalid_record = make_record("sample.png", 0)
    invalid_record["label"] = 2
    write_metadata(metadata_path, [invalid_record])

    with pytest.raises(ValueError, match="binary values"):
        ChestXrayDataset(metadata_path, image_dir)


def test_dataset_rejects_out_of_range_index(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "sample.png")
    metadata_path = tmp_path / "labels.csv"
    write_metadata(metadata_path, [make_record("sample.png", 0)])
    dataset = ChestXrayDataset(metadata_path, image_dir)

    with pytest.raises(IndexError):
        dataset[1]


def test_patient_splits_are_disjoint_and_reproducible():
    records = [
        {
            "Patient ID": patient_id,
            "Image Index": f"image_{patient_id}.png",
            "label": patient_id % 2,
        }
        for patient_id in range(1, 31)
    ]
    dataframe = pd.DataFrame(records)

    first = create_patient_splits(dataframe, random_state=17)
    second = create_patient_splits(dataframe, random_state=17)

    patient_sets = {
        name: set(split["Patient ID"])
        for name, split in first.items()
    }
    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])
    for name in ("train", "val", "test"):
        assert first[name]["Patient ID"].tolist() == second[name]["Patient ID"].tolist()


def test_patient_splits_reject_missing_patient_ids():
    dataframe = pd.DataFrame(
        {
            "Patient ID": [1, 2, None],
            "Image Index": ["a.png", "b.png", "c.png"],
            "label": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        create_patient_splits(dataframe)
