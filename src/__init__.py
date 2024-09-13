# from pathlib import Path

# from .models.my_model import MyModel
# from .models.resnet import Resnet
# from .models.vgg19 import VGG19

# model = Resnet()
# if not model.is_trained():
#     model.train()
# print(model.predict(Path("samples/1.jpg")))

import sys

from .gui.dialog import Dialog
from .gui.qt import *

app = QApplication(sys.argv)
dialog = Dialog()
dialog.show()
app.exec()
