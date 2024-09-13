from pathlib import Path

DATASET_PATH = Path("dataset")
TRAIN_PATH = DATASET_PATH / "train"
VALIDATION_PATH = DATASET_PATH / "validation"
TEST_PATH = DATASET_PATH / "test"
IMAGE_CATEGORIES = [p.name for p in TRAIN_PATH.iterdir()]
MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)


def get_images(type: str) -> list[Path]:
    images = []
    for category in (DATASET_PATH / type).iterdir():
        images.extend(list(category.iterdir()))
    return images
