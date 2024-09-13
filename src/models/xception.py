from tensorflow.keras.applications import Xception as XceptionKeras

from ..dataset import *
from .keras_model import KerasModel


class Xception(KerasModel):
    name = "xception"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model_class = XceptionKeras
