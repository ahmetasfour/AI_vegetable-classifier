from __future__ import annotations

from typing import Any

import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from ..dataset import *
from .model import Model


class MyModel(Model):
    name = "my_model"

    def train(self) -> Any:
        # Veri Çoklama
        train_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
        )
        train_image_generator = train_gen.flow_from_directory(
            TRAIN_PATH,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        val_gen = ImageDataGenerator()
        val_image_generator = val_gen.flow_from_directory(
            VALIDATION_PATH,
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        self.model = model = Sequential()

        model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding="Same",
                activation="relu",
                input_shape=(150, 150, 3),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(
            Conv2D(filters=96, kernel_size=(3, 3), padding="Same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(
            Conv2D(filters=96, kernel_size=(3, 3), padding="Same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(512, activation="relu"))

        model.add(Dense(len(self.category_names), activation="softmax"))

        model.summary()

        # Checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path / "checkpoint.hdf5",
            monitor="val_accuracy",
            verbose=2,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            patience=self.patience, monitor="val_accuracy", restore_best_weights=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            # Metrics: Accuracy, recall, etc
            metrics=[
                tf.metrics.Accuracy(),
                tf.metrics.Recall(),
                tf.metrics.Precision(),
                tf.metrics.SensitivityAtSpecificity(0.5),
                tf.metrics.SpecificityAtSensitivity(0.5),
            ],
        )
        hist = model.fit(
            train_image_generator,
            epochs=self.epochs,
            verbose=1,
            validation_data=val_image_generator,
            steps_per_epoch=len(get_images("train")) // self.batch_size,
            validation_steps=len(get_images("validation")) // self.batch_size,
            callbacks=[early_stopping, checkpoint],
        )
        model.save(self.model_path)

        return hist
