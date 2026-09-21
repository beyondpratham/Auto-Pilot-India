import numpy as np

from autopilot.config import IMAGE_SIZE
from autopilot.features import build_hog_descriptor, extract_hog_features


def test_extract_hog_features_shape():
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(4)]

    features = extract_hog_features(images, IMAGE_SIZE)
    descriptor_size = build_hog_descriptor(IMAGE_SIZE).getDescriptorSize()

    assert features.shape == (4, descriptor_size)
    assert features.dtype == np.float32


def test_extract_hog_features_resizes_mismatched_input():
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)]

    features = extract_hog_features(images, IMAGE_SIZE)
    descriptor_size = build_hog_descriptor(IMAGE_SIZE).getDescriptorSize()

    assert features.shape == (1, descriptor_size)
