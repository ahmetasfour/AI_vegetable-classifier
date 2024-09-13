from tensorflow.keras.applications import ResNet152V2 as RestNetKeras

from ..dataset import *
from .keras_model import KerasModel


class Resnet(KerasModel):
    name = "resnet"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model_class = RestNetKeras
