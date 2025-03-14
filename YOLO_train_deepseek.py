import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import yaml
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QGroupBox, QLineEdit, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QMessageBox, QLabel
)
from PySide6.QtCore import Qt
from ultralytics import YOLO


class YOLOv11TrainerPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11全参训练工作站 v3.0")
        self.setGeometry(100, 100, 1280, 800)

        self.init_ui()
        self.setup_layout()
        self.set_defaults()

    def init_ui(self):
        """初始化所有UI组件"""
        # 项目配置
        self.project_name = QLineEdit("yolov11_pro")
        self.save_dir = QLineEdit(str(Path.home() / "yolo_runs"))
        self.browse_save_btn = QPushButton("...", clicked=lambda: self.select_dir(self.save_dir))

        # 数据集配置
        self.data_dir = QLineEdit()
        self.browse_data_btn = QPushButton("...", clicked=lambda: self.select_dir_yaml(self.data_dir))
        self.nc = QSpinBox()
        self.nc.setRange(1, 1000)

        # 模型参数
        self.model_type = QComboBox()
        self.model_type.addItems(["n", "s", "m", "l", "x"])
        self.pretrained = QComboBox()
        self.pretrained.addItems(["none", "yolov11n.pt", "yolov11s.pt", "custom..."])
        self.depth = QDoubleSpinBox()
        self.depth.setRange(0.0, 1.0)
        self.width = QDoubleSpinBox()
        self.width.setRange(0.0, 1.0)

        # 优化器参数
        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW", "RMSprop"])
        self.lr0 = QDoubleSpinBox()
        self.lr0.setRange(0.0001, 1.0)
        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 0.99)
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 0.001)

        # 训练参数
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.batch = QSpinBox()
        self.batch.setRange(1, 256)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.patience = QSpinBox()
        self.patience.setRange(0, 100)

        # 数据增强
        self.fliplr = QDoubleSpinBox()
        self.fliplr.setRange(0.0, 1.0)
        self.hsv_h = QDoubleSpinBox()
        self.hsv_h.setRange(0.0, 0.1)
        self.hsv_s = QDoubleSpinBox()
        self.hsv_s.setRange(0.0, 0.5)
        self.hsv_v = QDoubleSpinBox()
        self.hsv_v.setRange(0.0, 0.5)
        self.mosaic = QCheckBox("马赛克增强")
        self.mixup = QCheckBox("MixUp增强")

        # 高级参数
        self.label_smoothing = QDoubleSpinBox()
        self.label_smoothing.setRange(0.0, 0.1)
        self.close_mosaic = QSpinBox()
        self.close_mosaic.setRange(-1, 100)
        self.box = QDoubleSpinBox()
        self.box.setRange(0.0, 10.0)
        self.cls = QDoubleSpinBox()
        self.cls.setRange(0.0, 10.0)
        self.dfl = QDoubleSpinBox()
        self.dfl.setRange(0.0, 10.0)

        # 功能按钮
        self.train_btn = QPushButton("开始训练", clicked=self.start_training)
        self.export_btn = QPushButton("导出配置", clicked=self.export_config)

    def setup_layout(self):
        """构建紧凑型布局"""
        main_widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(main_widget)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        main_layout = QVBoxLayout()

        # 项目配置组
        project_group = self.create_group("项目配置", [
            ("项目名称", self.project_name),
            ("保存路径", self.save_dir, self.browse_save_btn),
            ("数据集路径", self.data_dir, self.browse_data_btn),
            ("类别数", self.nc)
        ], cols=4)

        # 模型参数组
        model_group = self.create_group("模型参数", [
            ("模型类型", self.model_type),
            ("预训练权重", self.pretrained),
            ("深度系数", self.depth),
            ("宽度系数", self.width)
        ], cols=4)

        # 优化器组
        optim_group = self.create_group("优化参数", [
            ("优化器", self.optimizer),
            ("初始学习率", self.lr0),
            ("动量", self.momentum),
            ("权重衰减", self.weight_decay)
        ], cols=4)

        # 训练参数组
        train_group = self.create_group("训练参数", [
            ("训练轮次", self.epochs),
            ("批大小", self.batch),
            ("图像尺寸", self.imgsz),
            ("数据进程", self.workers),
            ("早停轮次", self.patience)
        ], cols=5)

        # 数据增强组
        augment_group = self.create_group("数据增强", [
            ("水平翻转", self.fliplr),
            ("色调增强", self.hsv_h),
            ("饱和度增强", self.hsv_s),
            ("亮度增强", self.hsv_v),
            (None, self.mosaic),
            (None, self.mixup)
        ], cols=6)

        # 高级参数组
        advance_group = self.create_group("高级参数", [
            ("标签平滑", self.label_smoothing),
            ("关闭马赛克", self.close_mosaic),
            ("Box权重", self.box),
            ("Cls权重", self.cls),
            ("DFL权重", self.dfl)
        ], cols=5)

        # 按钮组
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.train_btn)

        main_layout.addWidget(project_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(optim_group)
        main_layout.addWidget(train_group)
        main_layout.addWidget(augment_group)
        main_layout.addWidget(advance_group)
        main_layout.addLayout(btn_layout)
        main_widget.setLayout(main_layout)

    def create_group(self, title, widgets, cols=4):
        """创建参数分组"""
        group = QGroupBox(title)
        layout = QGridLayout()

        row = col = 0
        for widget in widgets:
            if widget[0] is not None:
                label = QLabel(widget[0])
                layout.addWidget(label, row, col * 2, 1, 1)
                if len(widget) > 2:
                    hbox = QHBoxLayout()
                    hbox.addWidget(widget[1])
                    hbox.addWidget(widget[2])
                    layout.addLayout(hbox, row, col * 2 + 1, 1, 1)
                else:
                    layout.addWidget(widget[1], row, col * 2 + 1, 1, 1)
            else:
                layout.addWidget(widget[1], row, col * 2, 1, 2)

            col += 1
            if col >= cols:
                col = 0
                row += 1

        group.setLayout(layout)
        return group

    def set_defaults(self):
        """设置默认参数"""
        defaults = {
            'nc': 80,
            'depth': 0.33,
            'width': 0.25,
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'workers': 8,
            'patience': 50,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'mosaic': True,
            'mixup': True,
            'label_smoothing': 0.1,
            'close_mosaic': 10,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        }
        for k, v in defaults.items():
            widget = getattr(self, k)
            if isinstance(widget, QCheckBox):
                widget.setChecked(v)
            else:
                widget.setValue(v)

    def select_dir(self, line_edit):
        """选择目录"""
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            line_edit.setText(path)

    def select_dir_yaml(self, line_edit):
        """选择目录"""
        path, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "文本文件 (*.yaml)")
        if path:
            line_edit.setText(path)

    def validate_inputs(self):
        """验证输入有效性"""
        required = [
            (self.data_dir, "数据集路径不能为空"),
            (self.save_dir, "保存路径不能为空"),
            (self.nc.value() > 0, "类别数必须大于0"),
            (Path(self.data_dir.text()).exists(), "数据集路径不存在")
        ]
        for condition, msg in required:
            if isinstance(condition, QLineEdit):
                if not condition.text().strip():
                    QMessageBox.critical(self, "错误", msg)
                    return False
            elif not condition:
                QMessageBox.critical(self, "错误", msg)
                return False
        return True

    def get_config(self):
        """获取训练配置"""
        return {
            'model': f'yolo11{self.model_type.currentText()}.pt' if self.pretrained.currentIndex() == 0 else self.pretrained.currentText(),
            'data': str(Path(self.data_dir.text())),
            'epochs': self.epochs.value(),
            'batch': self.batch.value(),
            'imgsz': self.imgsz.value(),
            'workers': self.workers.value(),
            'patience': self.patience.value(),
            'project': self.project_name.text(),
            'save_dir': self.save_dir.text(),
            'optimizer': self.optimizer.currentText(),
            'lr0': self.lr0.value(),
            'momentum': self.momentum.value(),
            'weight_decay': self.weight_decay.value(),
            'fliplr': self.fliplr.value(),
            'hsv_h': self.hsv_h.value(),
            'hsv_s': self.hsv_s.value(),
            'hsv_v': self.hsv_v.value(),
            'mosaic': self.mosaic.isChecked(),
            'mixup': self.mixup.isChecked(),
            'label_smoothing': self.label_smoothing.value(),
            'close_mosaic': self.close_mosaic.value(),
            'box': self.box.value(),
            'cls': self.cls.value(),
            'dfl': self.dfl.value()
        }

    def export_config(self):
        """导出配置文件"""
        if self.validate_inputs():
            config = self.get_config()
            path, _ = QFileDialog.getSaveFileName(self, "保存配置", "", "YAML Files (*.yaml)")
            if path:
                with open(path, 'w') as f:
                    yaml.dump(config, f)
                QMessageBox.information(self, "导出成功", "配置文件已保存")

    def start_training(self):
        """启动训练任务"""
        if not self.validate_inputs():
            return

        config = self.get_config()
        try:
            model = YOLO(config['model'])
            results = model.train(
                data=config['data'],
                epochs=config['epochs'],
                batch=config['batch'],
                imgsz=config['imgsz'],
                workers=config['workers'],
                patience=config['patience'],
                project=config['project'],
                optimizer=config['optimizer'],
                lr0=config['lr0'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                fliplr=config['fliplr'],
                hsv_h=config['hsv_h'],
                hsv_s=config['hsv_s'],
                hsv_v=config['hsv_v'],
                mosaic=config['mosaic'],
                mixup=config['mixup'],
                label_smoothing=config['label_smoothing'],
                close_mosaic=config['close_mosaic'],
                box=config['box'],
                cls=config['cls'],
                dfl=config['dfl']
            )
            QMessageBox.information(self, "训练完成", "模型训练成功完成！")
        except Exception as e:
            QMessageBox.critical(self, "训练错误", f"训练失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv11TrainerPro()
    window.show()
    sys.exit(app.exec())