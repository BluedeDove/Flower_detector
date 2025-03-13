import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 新增：防止PyTorch为GPU创建额外进程
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib  # 修复FigureCanvas错误

matplotlib.use('Qt5Agg')  # 设置Matplotlib后端


import re
import torch
from ultralytics import YOLO
import cv2
import gc
import threading

gc.collect()  # 强制垃圾收集器运行

import ui_app
import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QMessageBox
from PySide6.QtCore import QTimer, Signal, QObject
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage


class MemoryCleanerCallback:
    def __init__(self):
        self.epoch_count = 0

    def __call__(self, trainer):  # 添加__call__方法使对象可调用
        """每个epoch结束时触发内存清理"""
        self.epoch_count += 1

        # 清理PyTorch的显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 强制Python垃圾回收
        gc.collect()

        # 清理临时文件（需根据实际路径调整）
        #self._clean_temp_files(f"runs/detect/train{self.epoch_count}")

    def _clean_temp_files(self, temp_dir):
        """删除临时评估文件"""
        import os
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"清理临时文件: {temp_dir}")


# 创建用于线程通信的信号类
class TrainingSignals(QObject):
    progress = Signal(int)
    finished = Signal(bool, str)  # 成功状态, 消息
    update_image = Signal(str)  # 图像路径


class MainWindow(QMainWindow, ui_app.Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model_address.setVisible(False)
        self.model_save.setVisible(False)
        self.photo_selecting.setVisible(False)
        self.camera_select_comboBox.setVisible(False)
        self.mode_comboBox.setVisible(False)
        self.training_data_select.setVisible(False)
        self.start_tracking.setVisible(False)
        self.start_training_model.setVisible(False)
        self.epoch_select_Box.setVisible(False)

        self.model = None

        # mode_comboBox
        self.mode_comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        self.mode_comboBox.setCurrentIndex(-1)

        # camera_select_comboBox
        self.cap = None
        self.iscap_Ready = False
        for i in range(self.find_camera_num() + 1):
            showtext = f"摄像头{i}号"
            self.camera_select_comboBox.addItem(showtext)

        self.camera_select_comboBox.currentIndexChanged.connect(self.camera_select_change)
        self.camera_select_comboBox.setCurrentIndex(-1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.updata_frame)

        # model_select
        self.model_address.clicked.connect(self.select_model)
        self.ismodel_Ready = False
        self.model_path = None

        # photo_select
        self.photo_selecting.clicked.connect(self.select_photo)
        self.isphoto_Ready = False
        self.photo_frame = None

        # start_tracking
        '''try:
            self.start_tracking.clicked.disconnect()
        except RuntimeWarning:
            pass'''
        self.start_tracking.clicked.connect(self.start_tracking_it)

        # start_training
        self.start_training_model.clicked.connect(self.start_training)

        # training_model_save_path
        self.model_save.clicked.connect(self.select_save_path)
        self.ismodelsavepath_Ready = False
        self.model_save_path = None

        # training_datasets_path
        self.training_data_select.clicked.connect(self.select_training_datasets)
        self.isdatasets_Ready = False
        self.datasets = None

        # label
        self.show_me_image.clicked.connect(self.open_this_frame)
        self.this_frame = None
        self.isresult_Ready = False

        # cmd
        self.plaintextcmd_text_show.setStyleSheet("rgba(45, 45, 45, 150);"
                                                  "background-color: rgba(20, 20, 20, 0);")

        # 添加训练线程相关初始化
        self.training_signals = TrainingSignals()
        self.training_signals.progress.connect(self.update_progress)
        self.training_signals.finished.connect(self.training_finished)
        self.training_signals.update_image.connect(self.update_result_image)
        self.training_thread = None

        #sys.stdout = self
        #sys.stderr = self

    def on_combobox_changed(self, index):
        self.start_tracking.setText("启动识别")
        '''if self.start_tracking.isSignalConnected():
            self.start_tracking.clicked.disconnect()'''
        self.start_tracking.clicked.disconnect()
        self.start_tracking.clicked.connect(self.start_tracking_it)
        if self.mode_comboBox.isVisible():
            if index == 0:
                self.model_address.setVisible(True)
                self.model_save.setVisible(False)
                self.photo_selecting.setVisible(False)
                self.camera_select_comboBox.setVisible(True)
                self.training_data_select.setVisible(False)

            if index == 1:
                self.model_address.setVisible(True)
                self.model_save.setVisible(False)
                self.photo_selecting.setVisible(True)
                self.camera_select_comboBox.setVisible(False)
                self.training_data_select.setVisible(False)

        else:
            self.model_address.setVisible(False)
            self.photo_selecting.setVisible(False)
            self.camera_select_comboBox.setVisible(False)
            self.training_data_select.setVisible(False)

    def find_camera_num(self):
        total_num = 0
        while True:
            cap = cv2.VideoCapture(total_num, cv2.CAP_DSHOW)
            if cap.isOpened():  # 检查摄像头是否成功打开
                # yield total_num
                cap.release()
            else:
                break
            total_num += 1

        return total_num

    def show_warning(self, message, title="警告"):
        msg_box = QMessageBox.warning(None, title, message)

    def camera_select_change(self, index):
        # 如果之前有打开的摄像头，先释放它
        self.iscap_Ready = False

        if self.cap is not None:
            self.cap.release()

        # 如果索引为-1，表示没有选择任何摄像头
        if index < 0:
            self.cap = None
        else:
            # 创建新的摄像头对象
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

            # 检查摄像头是否成功打开
            if not self.cap.isOpened():
                # print(f"无法打开摄像头{index}")
                self.cap = None
                self.show_warning(f"无法打开摄像头{index}号")
            else:
                self.iscap_Ready = True
                print(f"摄像头选择：摄像头{index}号")

    def updata_frame(self):
        if self.camera_select_comboBox.isVisible():
            ret, frame = self.cap.read()
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(RGB_frame)

            for result in results:
                # 检查是否有检测到的对象
                if len(result.boxes.xyxy) == 0:
                    # print("No detections found.")
                    continue

                for i in range(result.boxes.xyxy.cpu().numpy().shape[0]):
                    detection = result.boxes.xyxy[i].cpu().numpy()

                    x1, y1, x2, y2 = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = result.boxes.conf[i].cpu().numpy()
                    if confidence < 0.5:
                        continue
                    class_id = result.boxes.cls[i].cpu().numpy()

                    class_name = result.names[int(class_id)]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.this_frame = frame
            self.isresult_Ready = True
            q_image = QImage(RGB_frame.data, RGB_frame.shape[1], RGB_frame.shape[0], RGB_frame.strides[0],
                             QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_image))
        pass

    def stop_tracking_it(self):
        self.start_tracking.setText("启动识别")
        self.start_tracking.clicked.disconnect()
        self.start_tracking.clicked.connect(self.start_tracking_it)
        self.timer.stop()
        self.plaintextcmd_text_show.setVisible(True)
        pass

    def start_tracking_it(self):
        if self.camera_select_comboBox.isVisible():
            if not self.iscap_Ready:
                self.show_warning("请选择可以使用的摄像头")

            if not self.ismodel_Ready:
                self.show_warning("请选择可以使用的模型")

            if self.iscap_Ready and self.ismodel_Ready:  # 写正式的处理流程
                print(f"开始识别\n摄像头：{self.camera_select_comboBox.currentIndex()}号\n"
                      f"模型地址：{self.model_path}")
                self.progressBar.setValue(0)
                self.model = YOLO("yolo11n.yaml")
                self.model = YOLO(self.model_path)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                self.progressBar.setValue(40)
                self.timer.start(30)
                self.progressBar.setValue(80)
                self.start_tracking.setText("点击停止")
                self.start_tracking.clicked.disconnect()
                self.start_tracking.clicked.connect(self.stop_tracking_it)
                self.plaintextcmd_text_show.setVisible(False)
                self.progressBar.setValue(100)

        if self.photo_selecting.isVisible():
            if not self.isphoto_Ready:
                self.show_warning("请选择要检测的图片")

            if not self.ismodel_Ready:
                self.show_warning("请选择可以使用的模型")

            if self.isphoto_Ready and self.ismodel_Ready:  # 写正式的处理流程
                print(f"开始识别\n图片地址：{self.photo_frame}\n"
                      f"模型地址：{self.model_path}")
                self.progressBar.setValue(0)
                self.model = YOLO("yolo11n.yaml")
                self.model = YOLO(self.model_path)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                self.progressBar.setValue(40)
                image = cv2.imread(self.photo_frame)
                RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.model(RGB_image)

                for result in results:
                    # 检查是否有检测到的对象
                    if len(result.boxes.xyxy) == 0:
                        print("图中没有检测到对象")
                        continue

                    for i in range(result.boxes.xyxy.cpu().numpy().shape[0]):
                        detection = result.boxes.xyxy[i].cpu().numpy()

                        x1, y1, x2, y2 = detection
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        confidence = result.boxes.conf[i].cpu().numpy()
                        '''if confidence < 0.5:
                            continue'''
                        class_id = result.boxes.cls[i].cpu().numpy()

                        class_name = result.names[int(class_id)]

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0), 2)
                RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.this_frame = image
                self.isresult_Ready = True
                self.progressBar.setValue(61)
                q_image = QImage(RGB_image.data, RGB_image.shape[1], RGB_image.shape[0], RGB_image.strides[0],
                                 QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(q_image))
                self.progressBar.setValue(100)
                pass

        if not self.camera_select_comboBox.isVisible() and not self.photo_selecting.isVisible():
            self.show_warning("请选择一个模式")

    # 新增：训练进度更新方法
    def update_progress(self, value):
        self.progressBar.setValue(value)

    # 新增：训练完成回调方法
    def training_finished(self, success, message):
        if success:
            self.show_warning("训练完成", "恭喜")
        else:
            self.show_warning(f"训练过程中发生错误: {message}")

        # 重新启用训练按钮
        self.start_training_model.setEnabled(True)

    # 新增：更新结果图像方法
    def update_result_image(self, image_path):
        if os.path.exists(image_path):
            result_image = cv2.imread(image_path)
            if result_image is not None:
                self.this_frame = result_image
                self.label.setPixmap(QPixmap(image_path))
                self.isresult_Ready = True
                # 在单独窗口显示结果
                cv2.imshow('results', result_image)
                cv2.waitKey(1)  # 非阻塞等待

    # 新增：训练线程方法
    def _train_model_thread(self):
        # 在此线程中创建新的模型实例
        model = YOLO("yolo11n.pt")

        # 配置设备
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        # device = torch.device('cpu')

        gc.collect()

        model.to(device)

        # 打印设备信息用于调试
        if use_cuda:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            torch.cuda.empty_cache()
        else:
            print("使用CPU训练")

        model.add_callback("on_train_epoch_end", MemoryCleanerCallback())



        this_name = os.path.splitext(os.path.basename(self.model_save_path))[0]

        self.training_signals.progress.emit(40)

        print(f"开始训练\n训练集地址：{self.datasets}\n训练周期：{self.epoch_select_Box.value()}\n"
              f"项目名称：{this_name}\n保存地址：{self.model_save_path}")

        # 使用特定GPU设置训练模型
        model.train(
            batch=-1,
            data=self.datasets,
            epochs=self.epoch_select_Box.value(),
            name=this_name,
            workers=0  # 禁用多进程工作器
        )

        self.training_signals.progress.emit(80)

        # 保存模型
        model.save(self.model_save_path)

        # 清理GPU内存
        if use_cuda:
            torch.cuda.empty_cache()

        # 检查结果图像
        open_image_path = os.path.join("runs", "detect", this_name, "results.png")
        if os.path.exists(open_image_path):
            self.training_signals.update_image.emit(open_image_path)
        else:
            # print(f"结果图像未找到: {open_image_path}")
            self.show_warning(f"结果图像未找到: {open_image_path}")

        self.training_signals.progress.emit(100)
        self.label.setPixmap(QPixmap(open_image_path))
        result_image = cv2.imread(open_image_path)
        cv2.imshow('results', result_image)
        self.training_signals.finished.emit(True, "训练成功完成")
        cv2.waitKey()
        '''try:
            # 在此线程中创建新的模型实例
            model = YOLO("yolo11n.pt")

            # 配置设备
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            #device = torch.device('cpu')

            gc.collect()

            # 打印设备信息用于调试
            if use_cuda:
                print(f"使用GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA版本: {torch.version.cuda}")
                torch.cuda.empty_cache()
            else:
                print("使用CPU训练")

            model.to(device)

            #model.add_callback("on_train_epoch_end", MemoryCleanerCallback())

            this_name = os.path.splitext(os.path.basename(self.model_save_path))[0]

            self.training_signals.progress.emit(40)

            print(f"开始训练\n训练集地址：{self.datasets}\n训练周期：{self.epoch_select_Box.value()}\n"
                  f"项目名称：{this_name}\n保存地址：{self.model_save_path}")

            # 使用特定GPU设置训练模型
            model.train(
                batch=32,
                data=self.datasets,
                epochs=self.epoch_select_Box.value(),
                name=this_name,
                workers=0  # 禁用多进程工作器
            )

            self.training_signals.progress.emit(80)

            # 保存模型
            model.save(self.model_save_path)

            # 清理GPU内存
            if use_cuda:
                torch.cuda.empty_cache()

            # 检查结果图像
            open_image_path = os.path.join("runs", "detect", this_name, "results.png")
            if os.path.exists(open_image_path):
                self.training_signals.update_image.emit(open_image_path)
            else:
                #print(f"结果图像未找到: {open_image_path}")
                self.show_warning(f"结果图像未找到: {open_image_path}")

            self.training_signals.progress.emit(100)
            self.label.setPixmap(QPixmap(open_image_path))
            result_image = cv2.imread(open_image_path)
            cv2.imshow('results', result_image)
            self.training_signals.finished.emit(True, "训练成功完成")
            cv2.waitKey()

        except Exception as exc:
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
            print(f"训练错误: {exc}")
            import traceback
            traceback.print_exc()
            self.training_signals.finished.emit(False, str(exc))'''

    # 修改：start_training方法，使用线程
    def start_training(self):
        if not self.ismodelsavepath_Ready:
            self.show_warning("请选择模型保存位置")
            return

        if not self.isdatasets_Ready:
            self.show_warning("请选择数据集配置文件")
            return

        if self.training_thread and self.training_thread.is_alive():
            self.show_warning("训练已在进行中")
            return

        # 禁用训练按钮防止多次点击
        self.start_training_model.setEnabled(False)

        # 重置进度
        self.progressBar.setValue(0)

        # 创建并启动训练线程
        self.training_thread = threading.Thread(target=self._train_model_thread)
        self.training_thread.daemon = True  # 允许应用关闭时终止线程
        self.training_thread.start()

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "文本文件 (*.pt)")
        if file_name:
            self.model_path = file_name
            self.ismodel_Ready = True
            print(f"选择模型：{file_name}")
        pass

    def select_photo(self):
        photo_file_name, _ = QFileDialog.getOpenFileName(None, "保存文件", "",
                                                         "所有图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)")
        if photo_file_name:
            self.photo_frame = photo_file_name
            self.isphoto_Ready = True
            print(f"选择图片：{self.photo_frame}")
        pass

    def select_save_path(self):
        save_file_name, _ = QFileDialog.getSaveFileName(None, "保存文件", "", "文本文件 (*.pt)")
        if save_file_name:
            self.model_save_path = save_file_name
            self.ismodelsavepath_Ready = True
            print(f"选择保存位置：{save_file_name}")
        pass

    def select_training_datasets(self):
        file_name, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "文本文件 (*.yaml)")
        if file_name:
            self.datasets = file_name
            self.isdatasets_Ready = True
            print(f"选择训练集：{file_name}")
        pass

    def open_this_frame(self):
        if self.isresult_Ready:
            cv2.imshow("当前结果", self.this_frame)
        else:
            self.show_warning("请先生成结果")

    def write(self, text):
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        finall_text = ansi_escape.sub('', text)
        self.plaintextcmd_text_show.appendPlainText(finall_text)

    def flush(self):
        pass


app = QApplication(sys.argv)
win = MainWindow()


def init_element():
    win.mode_comboBox.setCurrentIndex(-1)
    win.camera_select_comboBox.setCurrentIndex(-1)
    win.photo_frame = None
    win.epoch_select_Box.setVisible(False)
    win.label.setPixmap(QPixmap(u"background.png"))


def start():
    init_element()
    win.apply_model.setDisabled(True)
    win.training_mode.setDisabled(False)
    win.mode_comboBox.setVisible(True)
    win.model_save.setVisible(False)
    win.training_data_select.setVisible(False)
    win.start_tracking.setVisible(True)
    win.start_training_model.setVisible(False)
    print(f"当前模式：识别模式")


win.apply_model.clicked.connect(start)


def train():
    init_element()
    win.apply_model.setDisabled(False)
    win.training_mode.setDisabled(True)
    win.model_address.setVisible(False)
    win.photo_selecting.setVisible(False)
    win.camera_select_comboBox.setVisible(False)
    win.training_data_select.setVisible(False)
    win.mode_comboBox.setVisible(False)
    win.model_save.setVisible(True)
    win.training_data_select.setVisible(True)
    win.start_tracking.setVisible(False)
    win.start_training_model.setVisible(True)
    win.epoch_select_Box.setVisible(True)

    win.this_frame = None
    win.isresult_Ready = False

    print(f"当前模式：训练模式")


win.training_mode.clicked.connect(train)

win.show()
app.exec()

