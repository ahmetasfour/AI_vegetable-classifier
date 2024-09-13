import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from pyparsing import Any
from tensorflow import keras

from ..dataset import *
from .model import Model


class KerasModel(Model):
    def train(self) -> Any:
        super().train()

        for layer in self.base_model.layers:
            layer.trainable = False

        self.model = model = Sequential(
            [
                self.base_model,
                GlobalAveragePooling2D(),
                Flatten(),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.25),
                Dense(128, activation="relu"),
                Dense(15, activation="softmax"),
            ]
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path / "checkpoint.hdf5",
            monitor="accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )
        early_stopping = keras.callbacks.EarlyStopping(
            patience=self.patience,
            monitor="accuracy",
            restore_best_weights=True,
            min_delta=0.1,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=[
                tf.metrics.Accuracy(),
                tf.metrics.Recall(),
                tf.metrics.Precision(),
                tf.metrics.SensitivityAtSpecificity(0.5),
                tf.metrics.SpecificityAtSensitivity(0.5),
            ],
        )
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
