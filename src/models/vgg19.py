from tensorflow.keras.applications import VGG19 as VGG19Keras

from ..dataset import *
from .keras_model import KerasModel


class VGG19(KerasModel):
    name = "vgg19"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model_class = VGG19Keras
