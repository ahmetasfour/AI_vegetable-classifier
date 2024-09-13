from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from ..dataset import get_images
from ..models import MODELS, Model
from .qt import *


class Dialog(QDialog):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Window)
        icon_path = str(Path(__file__).parent / "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.test_list = QListWidget(self)
        self.train_list = QListWidget(self)
        self.validation_list = QListWidget(self)
        layout = QGridLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Test"), 0, 0)
        layout.addWidget(self.test_list, 1, 0)
        layout.addWidget(QLabel("Train"), 0, 1)
        layout.addWidget(self.train_list, 1, 1)
        layout.addWidget(QLabel("Validation"), 0, 2)
        layout.addWidget(self.validation_list, 1, 2)

        options_group = QGroupBox("")
        options_group_layout = QFormLayout()
        options_group.setLayout(options_group_layout)
        model_combo = self.model_combo = QComboBox(self)
        for model in MODELS:
            model_combo.addItem(model.name)
        options_group_layout.addRow("Model", model_combo)
        self.epochs_spinbox = epochs_spinbox = QSpinBox()
        epochs_spinbox.setMinimum(1)
        epochs_spinbox.setMaximum(20)
        epochs_spinbox.setValue(Model.default_epochs)
        options_group_layout.addRow("Epochs", epochs_spinbox)
        self.learning_rate_spinbox = learning_rate_spinbox = QDoubleSpinBox()
        learning_rate_spinbox.setDecimals(6)
        learning_rate_spinbox.setMaximum(1)
        learning_rate_spinbox.setValue(Model.default_learning_rate)
        options_group_layout.addRow("Learning rate", learning_rate_spinbox)
        self.batch_size_spinbox = batch_size_spinbox = QSpinBox()
        batch_size_spinbox.setMinimum(1)
        batch_size_spinbox.setMaximum(len(get_images("train")))
        batch_size_spinbox.setValue(Model.default_batch_size)
        options_group_layout.addRow("Batch size", batch_size_spinbox)
        self.early_stopping_patience = early_stopping_patience = QSpinBox()
        early_stopping_patience.setMinimum(0)
        early_stopping_patience.setMaximum(20)
        early_stopping_patience.setValue(Model.default_patience)
        layout.addWidget(options_group, 2, 0, 1, 3)

        # parameters_label = QLabel(
        #     f"Epochs: {Model.epochs}<br>Learning rate: {Model.learning_rate}<br>Batch size: {Model.batch_size}"
        # )
        # parameters_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # layout.addWidget(parameters_label, 2, 0, 1, 3)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, 3, 0, 1, 3)

        train_group = QGroupBox("Train")
        train_group_layout = QVBoxLayout()
        train_group.setLayout(train_group_layout)
        layout.addWidget(train_group, 4, 0, 1, 4)
        train_button = QPushButton("Train")
        train_button.clicked.connect(self.on_train)
        train_group_layout.addWidget(train_button)
        # self.training_graph_label = QLabel()
        # train_group_layout.addWidget(self.training_graph_label)

        categorize_group = QGroupBox("Categorize")
        layout.addWidget(categorize_group, 5, 0, 1, 3)
        categorize_group_layout = QVBoxLayout()
        categorize_group.setLayout(categorize_group_layout)
        upload_image_button = QPushButton("Upload")
        upload_image_button.clicked.connect(self.on_upload)
        categorize_group_layout.addWidget(upload_image_button)
        category_label = self.category_label = QLabel("Category: ")
        categorize_group_layout.addWidget(category_label)

        evaluate_group = QGroupBox("Evaluate")
        layout.addWidget(evaluate_group, 6, 0, 1, 3)
        evaluate_group_layout = QVBoxLayout()
        evaluate_group.setLayout(evaluate_group_layout)
        evaluate_button = QPushButton("Evaluate All Models")
        evaluate_group_layout.addWidget(evaluate_button)
        evaluate_button.clicked.connect(self.on_evaluate)
        self.evaluation_text_browser = evaluation_text_browser = QTextBrowser()
        evaluate_group_layout.addWidget(evaluation_text_browser)

        self._fill_list("test", self.test_list)
        self._fill_list("train", self.train_list)
        self._fill_list("validation", self.validation_list)
        self.test_list.itemClicked.connect(self.on_image_item_clicked)
        self.train_list.itemClicked.connect(self.on_image_item_clicked)
        self.validation_list.itemClicked.connect(self.on_image_item_clicked)
        # self.test_list.itemDoubleClicked.connect(self.on_image_item_double_clicked)
        # self.train_list.itemDoubleClicked.connect(self.on_image_item_double_clicked)
        # self.validation_list.itemDoubleClicked.connect(
        #     self.on_image_item_double_clicked
        # )

    def on_image_item_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        pixmap = QPixmap()
        pixmap.load(str(path))
        self.image_label.setPixmap(pixmap)
        self.perdict_image(path)

    def on_image_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        plt.figure()
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.title(item.text())
        plt.axis("off")
        plt.show()

    def _fill_list(self, type: str, listwidget: QListWidget) -> None:
        listwidget.clear()
        for image in get_images(type):
            # self.test_list.addItem()
            item = QListWidgetItem(f"{image.parent.name}/{image.name}")
            item.setData(Qt.ItemDataRole.UserRole, image)
            listwidget.addItem(item)

    def on_upload(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, caption="Image", filter="All files (*.*)"
        )
        if not file:
            return
        self.perdict_image(file)

    def init_selected_model(self) -> Model:
        return MODELS[self.model_combo.currentIndex()](
            epochs=self.epochs_spinbox.value(),
            learning_rate=self.learning_rate_spinbox.value(),
            batch_size=self.batch_size_spinbox.value(),
            patience=self.early_stopping_patience.value(),
        )

    def perdict_image(self, file: str | Path) -> None:
        model = self.init_selected_model()
        category = model.predict(file)
        self.category_label.setText(f"Category: {category}")

    def on_evaluate(self) -> None:
        self.evaluation_text_browser.clear()
        for model_class in MODELS:
            model = model_class()
            evaluation = model.evaluate()
            text = f"{model.name}: " + " ".join(
                f"{k}={v}" for k, v in evaluation.items()
            )
            self.evaluation_text_browser.append(text)

    def on_train(self) -> None:
        model = self.init_selected_model()
        history = model.train().history
        print(f"{history=}")
        plt.style.use("ggplot")
        plt.figure(figsize=(10, 5))
        for label, value in history.items():
            plt.plot(value, label=label)

        # plt.plot(history["loss"], c="red", label="Loss")
        # plt.plot(history["accuracy"], c="orange", linestyle="--", label="Accuracy")
        # try:
        #     recall = history["recall"]
        # except KeyError:
        #     recall = history["recall_4"]
        # plt.plot(
        #     recall,
        #     c="yellow",
        #     label="Recall",
        # )
        # plt.plot(history["precision"], c="green", linestyle="--", label="Precision")
        # plt.plot(history["sensitivity_at_specificity"], c="blue", label="Sensitivity")
        # plt.plot(
        #     history["specificity_at_sensitivity"],
        #     c="indigo",
        #     linestyle="--",
        #     label="Specificity",
        # )
        plt.xlabel("Number of Epochs")
        plt.legend(loc="best")
        plt.show()

        # buf = BytesIO()
        # plt.savefig(buf)
        # buf.seek(0)
        # pixmap = QPixmap()
        # pixmap.loadFromData(buf.read())
        # self.training_graph_label.setPixmap(pixmap)
