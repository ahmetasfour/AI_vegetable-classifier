from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import numpy as np
import tensorflow as tf
from keras.models import Model as KerasModel
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from ..dataset import *


class Model:
    name: str
    default_epochs = 2
    default_learning_rate = 0.001
    default_batch_size = 16
    default_patience = 2

    def __init__(
        self,
        epochs: int = default_epochs,
        learning_rate: float = default_learning_rate,
        batch_size: int = default_batch_size,
        patience: int = default_patience,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.model_path = MODELS_PATH / self.name
        self.model: KerasModel | None = None
        self.base_model_class: Type[KerasModel] | None = None
        self.base_model: KerasModel | None = None
        if self.model_path.exists():
            self.model = tf.keras.models.load_model(self.model_path)
        test_gen = ImageDataGenerator()

        self.test_image_generator = test_gen.flow_from_directory(
            TEST_PATH,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def category_names(self) -> list[str]:
        return [p.name.replace("_", " ") for p in TRAIN_PATH.iterdir()]

    def train(self) -> Any:
        if self.base_model_class:
            self.base_model = self.base_model_class(
                weights="imagenet", include_top=False, input_shape=(150, 150, 3)
            )

    def predict(self, image_path: str | Path) -> str:
        if not self.is_trained():
            self.train()

        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        test_img_input = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1], img_array.shape[2])
        )
        return self.category_names[np.argmax(self.model.predict(test_img_input))]

    def evaluate(self) -> dict:
        if not self.is_trained():
            self.train()
        return self.model.evaluate(self.test_image_generator, return_dict=True)
