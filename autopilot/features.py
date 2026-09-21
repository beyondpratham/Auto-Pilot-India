import cv2
import numpy as np

from autopilot.config import (
    HOG_BLOCK_SIZE,
    HOG_BLOCK_STRIDE,
    HOG_CELL_SIZE,
    HOG_NBINS,
    IMAGE_SIZE,
)


def build_hog_descriptor(win_size=IMAGE_SIZE):
    return cv2.HOGDescriptor(
        win_size, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE, HOG_CELL_SIZE, HOG_NBINS
    )


def extract_hog_features(images, win_size=IMAGE_SIZE):
    hog = build_hog_descriptor(win_size)
    expected_hw = (win_size[1], win_size[0])
    features = np.empty((len(images), hog.getDescriptorSize()), dtype=np.float32)
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.shape[:2] != expected_hw:
            gray = cv2.resize(gray, win_size)
        features[i] = hog.compute(gray).flatten()
    return features
