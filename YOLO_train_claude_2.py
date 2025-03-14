import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QFileDialog,
                               QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt
from ultralytics import YOLO
class YOLOv11TrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11 Trainer")
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        # Project Information
        project_group = QGroupBox("Project Information")
        project_layout = QVBoxLayout()
        project_group.setLayout(project_layout)

        self.project_name = QLineEdit("YOLOv11_Project")
        project_layout.addWidget(QLabel("Project Name:"))
        project_layout.addWidget(self.project_name)

        self.experiment_name = QLineEdit("exp")
        project_layout.addWidget(QLabel("Experiment Name:"))
        project_layout.addWidget(self.experiment_name)

        # Data Selection
        self.data_yaml_path = QLineEdit()
        self.data_yaml_btn = QPushButton("Select Data YAML File")
        self.data_yaml_btn.clicked.connect(self.select_data_yaml)
        project_layout.addWidget(QLabel("Data YAML File:"))
        project_layout.addWidget(self.data_yaml_path)
        project_layout.addWidget(self.data_yaml_btn)

        # Output Directory
        self.output_dir = QLineEdit()
        self.output_dir_btn = QPushButton("Select Output Directory")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        project_layout.addWidget(QLabel("Output Directory:"))
        project_layout.addWidget(self.output_dir)
        project_layout.addWidget(self.output_dir_btn)

        # Pretrained Weights
        self.weights_path = QLineEdit()
        self.weights_path_btn = QPushButton("Select Pretrained Weights (Optional)")
        self.weights_path_btn.clicked.connect(self.select_weights)
        project_layout.addWidget(QLabel("Pretrained Weights:"))
        project_layout.addWidget(self.weights_path)
        project_layout.addWidget(self.weights_path_btn)

        scroll_layout.addWidget(project_group)

        # Model Structure Parameters
        structure_group = QGroupBox("Model Structure Parameters")
        structure_layout = QVBoxLayout()
        structure_group.setLayout(structure_layout)

        self.model_type = QComboBox()
        self.model_type.addItems(["YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x", "Custom"])
        self.model_type.currentIndexChanged.connect(self.update_model_params)
        structure_layout.addWidget(QLabel("Model Type:"))
        structure_layout.addWidget(self.model_type)

        self.depth_multiple = QDoubleSpinBox()
        self.depth_multiple.setRange(0.1, 1.0)
        self.depth_multiple.setValue(0.33)
        self.depth_multiple.setDecimals(2)
        structure_layout.addWidget(QLabel("Depth Multiple:"))
        structure_layout.addWidget(self.depth_multiple)

        self.width_multiple = QDoubleSpinBox()
        self.width_multiple.setRange(0.1, 1.0)
        self.width_multiple.setValue(0.25)
        self.width_multiple.setDecimals(2)
        structure_layout.addWidget(QLabel("Width Multiple:"))
        structure_layout.addWidget(self.width_multiple)

        self.max_channels = QSpinBox()
        self.max_channels.setRange(64, 2048)
        self.max_channels.setValue(1024)
        structure_layout.addWidget(QLabel("Max Channels:"))
        structure_layout.addWidget(self.max_channels)

        self.nc = QSpinBox()
        self.nc.setRange(1, 1000)
        self.nc.setValue(80)
        structure_layout.addWidget(QLabel("Number of Classes:"))
        structure_layout.addWidget(self.nc)

        scroll_layout.addWidget(structure_group)

        # Training Hyperparameters
        hyper_group = QGroupBox("Training Hyperparameters")
        hyper_layout = QVBoxLayout()
        hyper_group.setLayout(hyper_layout)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        hyper_layout.addWidget(QLabel("Epochs:"))
        hyper_layout.addWidget(self.epochs)

        self.batch = QSpinBox()
        self.batch.setRange(1, 256)
        self.batch.setValue(16)
        hyper_layout.addWidget(QLabel("Batch Size:"))
        hyper_layout.addWidget(self.batch)

        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setValue(640)
        self.imgsz.setSingleStep(32)
        hyper_layout.addWidget(QLabel("Image Size:"))
        hyper_layout.addWidget(self.imgsz)

        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setValue(8)
        hyper_layout.addWidget(QLabel("Workers:"))
        hyper_layout.addWidget(self.workers)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW", "RMSprop"])
        hyper_layout.addWidget(QLabel("Optimizer:"))
        hyper_layout.addWidget(self.optimizer)

        self.lr0 = QDoubleSpinBox()
        self.lr0.setRange(0.0001, 0.1)
        self.lr0.setValue(0.01)
        self.lr0.setDecimals(4)
        hyper_layout.addWidget(QLabel("Initial Learning Rate:"))
        hyper_layout.addWidget(self.lr0)

        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.8, 0.98)
        self.momentum.setValue(0.937)
        self.momentum.setDecimals(3)
        hyper_layout.addWidget(QLabel("Momentum:"))
        hyper_layout.addWidget(self.momentum)

        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0001, 0.001)
        self.weight_decay.setValue(0.0005)
        self.weight_decay.setDecimals(4)
        hyper_layout.addWidget(QLabel("Weight Decay:"))
        hyper_layout.addWidget(self.weight_decay)

        scroll_layout.addWidget(hyper_group)

        # Data Augmentation Parameters
        augment_group = QGroupBox("Data Augmentation Parameters")
        augment_layout = QVBoxLayout()
        augment_group.setLayout(augment_layout)

        self.hsv_h = QDoubleSpinBox()
        self.hsv_h.setRange(0, 0.1)
        self.hsv_h.setValue(0.015)
        self.hsv_h.setDecimals(3)
        augment_layout.addWidget(QLabel("HSV Hue:"))
        augment_layout.addWidget(self.hsv_h)

        self.hsv_s = QDoubleSpinBox()
        self.hsv_s.setRange(0, 0.5)
        self.hsv_s.setValue(0.7)
        self.hsv_s.setDecimals(3)
        augment_layout.addWidget(QLabel("HSV Saturation:"))
        augment_layout.addWidget(self.hsv_s)

        self.hsv_v = QDoubleSpinBox()
        self.hsv_v.setRange(0, 0.5)
        self.hsv_v.setValue(0.4)
        self.hsv_v.setDecimals(3)
        augment_layout.addWidget(QLabel("HSV Value:"))
        augment_layout.addWidget(self.hsv_v)

        self.fliplr = QDoubleSpinBox()
        self.fliplr.setRange(0, 1)
        self.fliplr.setValue(0.5)
        self.fliplr.setDecimals(2)
        augment_layout.addWidget(QLabel("Horizontal Flip Probability:"))
        augment_layout.addWidget(self.fliplr)

        self.mosaic = QCheckBox("Enable Mosaic")
        self.mosaic.setChecked(True)
        augment_layout.addWidget(self.mosaic)

        self.mixup = QCheckBox("Enable MixUp")
        self.mixup.setChecked(True)
        augment_layout.addWidget(self.mixup)

        scroll_layout.addWidget(augment_group)

        # Advanced Training Configuration
        advanced_group = QGroupBox("Advanced Training Configuration")
        advanced_layout = QVBoxLayout()
        advanced_group.setLayout(advanced_layout)

        self.resume = QCheckBox("Resume Training")
        advanced_layout.addWidget(self.resume)

        self.save_period = QSpinBox()
        self.save_period.setRange(1, 100)
        self.save_period.setValue(10)
        advanced_layout.addWidget(QLabel("Save Period (epochs):"))
        advanced_layout.addWidget(self.save_period)

        self.close_mosaic = QSpinBox()
        self.close_mosaic.setRange(0, 100)
        self.close_mosaic.setValue(10)
        advanced_layout.addWidget(QLabel("Close Mosaic (last N epochs):"))
        advanced_layout.addWidget(self.close_mosaic)

        self.label_smoothing = QDoubleSpinBox()
        self.label_smoothing.setRange(0, 0.1)
        self.label_smoothing.setValue(0.0)
        self.label_smoothing.setDecimals(3)
        advanced_layout.addWidget(QLabel("Label Smoothing:"))
        advanced_layout.addWidget(self.label_smoothing)

        self.device = QComboBox()
        self.device.addItems(["", "cpu", "0", "0,1", "0,1,2,3", "cuda"])
        advanced_layout.addWidget(QLabel("Device (empty for auto-detect):"))
        advanced_layout.addWidget(self.device)

        self.cache = QCheckBox("Cache Images in RAM")
        advanced_layout.addWidget(self.cache)

        self.multi_scale = QCheckBox("Multi-Scale Training")
        advanced_layout.addWidget(self.multi_scale)

        scroll_layout.addWidget(advanced_group)

        # Train Button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;")
        self.train_btn.clicked.connect(self.start_training)
        main_layout.addWidget(self.train_btn)

        # Initialize model parameters based on default selection
        self.update_model_params(0)

    def select_data_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data YAML File", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.data_yaml_path.setText(file_path)

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir.setText(folder)

    def select_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Pretrained Weights", "", "Model Files (*.pt *.pth)")
        if file_path:
            self.weights_path.setText(file_path)

    def update_model_params(self, index):
        # Set parameters based on model type
        if index == 0:  # YOLOv11n
            self.depth_multiple.setValue(0.33)
            self.width_multiple.setValue(0.25)
            self.max_channels.setValue(1024)
        elif index == 1:  # YOLOv11s
            self.depth_multiple.setValue(0.33)
            self.width_multiple.setValue(0.50)
            self.max_channels.setValue(1024)
        elif index == 2:  # YOLOv11m
            self.depth_multiple.setValue(0.67)
            self.width_multiple.setValue(0.75)
            self.max_channels.setValue(1024)
        elif index == 3:  # YOLOv11l
            self.depth_multiple.setValue(1.0)
            self.width_multiple.setValue(1.0)
            self.max_channels.setValue(1024)
        elif index == 4:  # YOLOv11x
            self.depth_multiple.setValue(1.33)
            self.width_multiple.setValue(1.25)
            self.max_channels.setValue(1280)
        # For "Custom", don't change values

    def start_training(self):
        # Validate inputs
        if not self.data_yaml_path.text():
            QMessageBox.warning(self, "Input Error", "Please select a data YAML file.")
            return

        if not self.output_dir.text():
            QMessageBox.warning(self, "Input Error", "Please select an output directory.")
            return

        # Collect all parameters
        params = {
            # Project settings
            'project': self.output_dir.text(),
            'name': self.experiment_name.text(),
            'data': self.data_yaml_path.text(),

            # Model structure
            'model': self.weights_path.text() if self.weights_path.text() else None,

            # Training hyperparameters
            'epochs': self.epochs.value(),
            'batch': self.batch.value(),
            'imgsz': self.imgsz.value(),
            'workers': self.workers.value(),
            'optimizer': self.optimizer.currentText(),
            'lr0': self.lr0.value(),
            'momentum': self.momentum.value(),
            'weight_decay': self.weight_decay.value(),

            # Data augmentation
            'hsv_h': self.hsv_h.value(),
            'hsv_s': self.hsv_s.value(),
            'hsv_v': self.hsv_v.value(),
            'fliplr': self.fliplr.value(),
            'mosaic': self.mosaic.isChecked(),
            'mixup': self.mixup.isChecked(),

            # Advanced configuration
            'resume': self.resume.isChecked(),
            'save_period': self.save_period.value(),
            'close_mosaic': self.close_mosaic.value(),
            'label_smoothing': self.label_smoothing.value(),
            'cache': self.cache.isChecked(),
            'multi_scale': self.multi_scale.isChecked(),
        }

        # Add device only if specified
        if self.device.currentText():
            params['device'] = self.device.currentText()

        try:
            # Create model config based on selected type
            model_type = self.model_type.currentText()
            if model_type == "Custom":
                # For custom, use the specific parameters
                model_config = {
                    'nc': self.nc.value(),
                    'depth_multiple': self.depth_multiple.value(),
                    'width_multiple': self.width_multiple.value(),
                    'max_channels': self.max_channels.value()
                }

                # Initialize and train YOLO model with custom config
                model = YOLO('yolo11n.pt', model_config)
            else:
                # Use predefined model type
                model_yaml = f"yolo11{model_type[-1].lower()}.pt"  # e.g., "yolov11n.yaml"
                model = YOLO(model_yaml)

            # Start training
            QMessageBox.information(self, "Training Started",
                                    f"Training has started with {self.epochs.value()} epochs.\n"
                                    f"Project: {self.project_name.text()}\n"
                                    f"Experiment: {self.experiment_name.text()}\n"
                                    f"Results will be saved to: {os.path.join(self.output_dir.text(), self.experiment_name.text())}")

            model.train(**params)

            QMessageBox.information(self, "Training Complete",
                                    "Training has completed successfully.\n"
                                    f"Results are saved to: {os.path.join(self.output_dir.text(), self.experiment_name.text())}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during training:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv11TrainerGUI()
    window.show()
    sys.exit(app.exec())
