import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from autopilot.config import IMAGE_SIZE, RANDOM_STATE

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ImageDataset:
    images: np.ndarray
    labels: np.ndarray
    class_names: list


def load_image_dataset(root_dir, image_size=IMAGE_SIZE):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Dataset directory '{root_dir}' does not exist. "
            "Run scripts/generate_demo_dataset.py or point --data at a real "
            "class-per-folder dataset (see data/README.md)."
        )

    class_names = sorted(
        entry.name for entry in os.scandir(root_dir) if entry.is_dir()
    )
    if not class_names:
        raise ValueError(f"No class subfolders found under '{root_dir}'.")

    images, labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        for entry in os.scandir(class_dir):
            if not entry.is_file():
                continue
            if os.path.splitext(entry.name)[1].lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(entry.path) as img:
                    img = img.convert("RGB").resize(image_size)
                    images.append(np.array(img, dtype=np.uint8))
                    labels.append(label_idx)
            except (UnidentifiedImageError, OSError):
                continue

    if not images:
        raise ValueError(f"No readable images found under '{root_dir}'.")

    return ImageDataset(
        images=np.stack(images),
        labels=np.array(labels, dtype=np.int64),
        class_names=class_names,
    )


def train_test_split_dataset(dataset: ImageDataset, test_size=0.2, random_state=RANDOM_STATE):
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.images,
        dataset.labels,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )
    return x_train, x_test, y_train, y_test
