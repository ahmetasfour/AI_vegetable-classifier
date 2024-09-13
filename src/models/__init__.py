from typing import Type

from .model import Model
from .my_model import MyModel
from .resnet import Resnet
from .vgg19 import VGG19
from .xception import Xception

MODELS: list[Type[Model]] = [MyModel, VGG19, Xception, Resnet]
