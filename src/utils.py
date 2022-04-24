from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np


def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"


def load_encoder(encoder_classes_path) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_classes_path, allow_pickle=True)

    return encoder
