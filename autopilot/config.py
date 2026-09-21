from dataclasses import dataclass


CLASS_NAMES = ["animal", "human", "nothing", "pothole", "vehicle"]

RISK_WEIGHTS = {
    "human": 0.15,
    "vehicle": 0.10,
    "animal": 0.05,
    "pothole": 0.02,
    "nothing": 0.0,
}

COCO_TO_RISK_CLASS = {
    "person": "human",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
}

IMAGE_SIZE = (128, 128)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

BRAKE_RISK_THRESHOLD = 0.12
NUM_SEGMENTS = 3
SEGMENT_DIRECTIONS = ["Go Left", "Go Straight", "Go Right"]
DETECTION_SCORE_THRESHOLD = 0.5

RANDOM_STATE = 42


@dataclass
class TrainingConfig:
    data_dir: str = "data/real"
    image_size: tuple = IMAGE_SIZE
    test_size: float = 0.2
    backbone: str = "vgg16"
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    output_dir: str = "outputs"
    random_state: int = RANDOM_STATE
