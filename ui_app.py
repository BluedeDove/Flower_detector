# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'app.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QLabel,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy,
    QSpinBox, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.setEnabled(True)
        Form.resize(957, 554)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QSize(957, 554))
        Form.setMaximumSize(QSize(957, 554))
        Form.setMouseTracking(False)
        Form.setAcceptDrops(False)
        icon = QIcon(QIcon.fromTheme(u"applications-development"))
        Form.setWindowIcon(icon)
        Form.setAutoFillBackground(False)
        Form.setInputMethodHints(Qt.InputMethodHint.ImhNone)
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(60, 40, 641, 391))
        self.label.setAcceptDrops(False)
        self.label.setAutoFillBackground(False)
        self.label.setTextFormat(Qt.TextFormat.AutoText)
        self.label.setPixmap(QPixmap(u"background.png"))
        self.label.setScaledContents(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.apply_model = QPushButton(Form)
        self.apply_model.setObjectName(u"apply_model")
        self.apply_model.setEnabled(True)
        self.apply_model.setGeometry(QRect(750, 40, 161, 81))
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSend))
        self.apply_model.setIcon(icon1)
        self.apply_model.setAutoDefault(False)
        self.training_mode = QPushButton(Form)
        self.training_mode.setObjectName(u"training_mode")
        self.training_mode.setEnabled(True)
        self.training_mode.setGeometry(QRect(750, 130, 161, 81))
        self.training_mode.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        icon2 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.AppointmentNew))
        self.training_mode.setIcon(icon2)
        self.mode_comboBox = QComboBox(Form)
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.setObjectName(u"mode_comboBox")
        self.mode_comboBox.setGeometry(QRect(750, 440, 161, 41))
        self.mode_comboBox.setEditable(False)
        self.progressBar = QProgressBar(Form)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(100, 450, 571, 101))
        self.progressBar.setValue(100)
        self.progressBar.setTextVisible(True)
        self.progressBar.setOrientation(Qt.Orientation.Horizontal)
        self.progressBar.setInvertedAppearance(False)
        self.model_address = QPushButton(Form)
        self.model_address.setObjectName(u"model_address")
        self.model_address.setEnabled(True)
        self.model_address.setGeometry(QRect(750, 340, 161, 41))
        icon3 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.AddressBookNew))
        self.model_address.setIcon(icon3)
        self.model_address.setAutoDefault(False)
        self.camera_select_comboBox = QComboBox(Form)
        self.camera_select_comboBox.setObjectName(u"camera_select_comboBox")
        self.camera_select_comboBox.setGeometry(QRect(750, 390, 161, 41))
        self.photo_selecting = QPushButton(Form)
        self.photo_selecting.setObjectName(u"photo_selecting")
        self.photo_selecting.setGeometry(QRect(750, 390, 161, 41))
        icon4 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.ListAdd))
        self.photo_selecting.setIcon(icon4)
        self.photo_selecting.setAutoDefault(False)
        self.model_save = QPushButton(Form)
        self.model_save.setObjectName(u"model_save")
        self.model_save.setEnabled(True)
        self.model_save.setGeometry(QRect(750, 340, 161, 41))
        self.model_save.setIcon(icon3)
        self.model_save.setAutoDefault(False)
        self.training_data_select = QPushButton(Form)
        self.training_data_select.setObjectName(u"training_data_select")
        self.training_data_select.setGeometry(QRect(750, 390, 161, 41))
        self.training_data_select.setIcon(icon4)
        self.training_data_select.setAutoDefault(False)
        self.start_tracking = QPushButton(Form)
        self.start_tracking.setObjectName(u"start_tracking")
        self.start_tracking.setEnabled(True)
        self.start_tracking.setGeometry(QRect(750, 250, 161, 41))
        icon5 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaybackStart))
        self.start_tracking.setIcon(icon5)
        self.start_tracking.setAutoDefault(False)
        self.start_training_model = QPushButton(Form)
        self.start_training_model.setObjectName(u"start_training_model")
        self.start_training_model.setEnabled(True)
        self.start_training_model.setGeometry(QRect(750, 250, 161, 41))
        self.start_training_model.setIcon(icon5)
        self.start_training_model.setAutoDefault(False)
        self.epoch_select_Box = QSpinBox(Form)
        self.epoch_select_Box.setObjectName(u"epoch_select_Box")
        self.epoch_select_Box.setGeometry(QRect(750, 440, 161, 41))
        self.epoch_select_Box.setWrapping(False)
        self.epoch_select_Box.setFrame(True)
        self.epoch_select_Box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.epoch_select_Box.setMaximum(25565)
        self.epoch_select_Box.setValue(50)
        self.show_me_image = QPushButton(Form)
        self.show_me_image.setObjectName(u"show_me_image")
        self.show_me_image.setGeometry(QRect(20, 440, 31, 31))
        icon6 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.ZoomIn))
        self.show_me_image.setIcon(icon6)
        self.plaintextcmd_text_show = QPlainTextEdit(Form)
        self.plaintextcmd_text_show.setObjectName(u"plaintextcmd_text_show")
        self.plaintextcmd_text_show.setGeometry(QRect(60, 40, 641, 391))
        font = QFont()
        font.setBold(True)
        font.setItalic(False)
        self.plaintextcmd_text_show.setFont(font)
        self.plaintextcmd_text_show.viewport().setProperty(u"cursor", QCursor(Qt.CursorShape.CrossCursor))
        self.plaintextcmd_text_show.setAcceptDrops(False)
        self.plaintextcmd_text_show.setAutoFillBackground(False)
        self.plaintextcmd_text_show.setInputMethodHints(Qt.InputMethodHint.ImhMultiLine)
        self.plaintextcmd_text_show.setFrameShape(QFrame.Shape.Box)
        self.plaintextcmd_text_show.setUndoRedoEnabled(False)
        self.plaintextcmd_text_show.setReadOnly(True)
        self.plaintextcmd_text_show.setBackgroundVisible(False)
        self.apply_model.raise_()
        self.training_mode.raise_()
        self.mode_comboBox.raise_()
        self.progressBar.raise_()
        self.label.raise_()
        self.model_address.raise_()
        self.camera_select_comboBox.raise_()
        self.photo_selecting.raise_()
        self.model_save.raise_()
        self.training_data_select.raise_()
        self.start_tracking.raise_()
        self.start_training_model.raise_()
        self.epoch_select_Box.raise_()
        self.show_me_image.raise_()
        self.plaintextcmd_text_show.raise_()

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u82b1\u5349\u8bc6\u522b\u5de5\u5177", None))
#if QT_CONFIG(statustip)
        Form.setStatusTip("")
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        Form.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(accessibility)
        Form.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
        self.label.setText("")
#if QT_CONFIG(whatsthis)
        self.apply_model.setWhatsThis(QCoreApplication.translate("Form", u"\u5f00\u59cb\u8bc6\u522b\u82b1\u5349", None))
#endif // QT_CONFIG(whatsthis)
        self.apply_model.setText(QCoreApplication.translate("Form", u"\u542f\u7528\u6a21\u578b", None))
#if QT_CONFIG(whatsthis)
        self.training_mode.setWhatsThis(QCoreApplication.translate("Form", u"\u8bad\u7ec3\u4f60\u7684\u82b1\u5349\u8bc6\u522b\u6a21\u578b", None))
#endif // QT_CONFIG(whatsthis)
        self.training_mode.setText(QCoreApplication.translate("Form", u"\u8bad\u7ec3\u6a21\u578b", None))
        self.mode_comboBox.setItemText(0, QCoreApplication.translate("Form", u"\u76f8\u673a\u6a21\u5f0f", None))
        self.mode_comboBox.setItemText(1, QCoreApplication.translate("Form", u"\u56fe\u7247\u6a21\u5f0f", None))

        self.progressBar.setFormat(QCoreApplication.translate("Form", u"%p%", None))
#if QT_CONFIG(whatsthis)
        self.model_address.setWhatsThis(QCoreApplication.translate("Form", u"\u9009\u62e9\u4f60\u7684\u6a21\u578b\uff0c\u82e5\u65e0\u5219\u4e3a\u9ed8\u8ba4\uff08\u57fa\u7840YOLO\u6a21\u578b\uff09", None))
#endif // QT_CONFIG(whatsthis)
        self.model_address.setText(QCoreApplication.translate("Form", u"\u6a21\u578b\u8def\u5f84", None))
#if QT_CONFIG(whatsthis)
        self.photo_selecting.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.photo_selecting.setText(QCoreApplication.translate("Form", u"\u9009\u62e9\u56fe\u7247", None))
#if QT_CONFIG(whatsthis)
        self.model_save.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.model_save.setText(QCoreApplication.translate("Form", u"\u4fdd\u5b58\u8def\u5f84", None))
#if QT_CONFIG(whatsthis)
        self.training_data_select.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.training_data_select.setText(QCoreApplication.translate("Form", u"\u9009\u62e9\u8bad\u7ec3\u96c6", None))
#if QT_CONFIG(whatsthis)
        self.start_tracking.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.start_tracking.setText(QCoreApplication.translate("Form", u"\u542f\u52a8\u8bc6\u522b", None))
#if QT_CONFIG(whatsthis)
        self.start_training_model.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.start_training_model.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb\u8bad\u7ec3", None))
        self.show_me_image.setText("")
        self.plaintextcmd_text_show.setPlainText(QCoreApplication.translate("Form", u"    ====================================\n"
"    |                                                                                          |\n"
"    |       \u82b1\u5349\u8bc6\u522b\u4e0e\u8ffd\u8e2a\u7cfb\u7edf - \u6df1\u5ea6\u5b66\u4e60\u7ec42025\u5047\u671f\u9879\u76ee           |\n"
"    |                                                                                          |\n"
"    ====================================\n"
"    \u4f5c\u8005: 24\u7269\u8054\u7f51\u5218\u627f\u6d69\n"
"    \u521b\u5efa\u65e5\u671f: 2025/1/22\n"
"    -----------------------------------------------------------------\n"
"    \u672c\u9879\u76ee\u5305\u62ec\u4ee5\u4e0b\u529f\u80fd:\n"
"    - \u6a21\u578b\u8bad\u7ec3\n"
"    - \u5355\u6444\u50cf\u5934\u82b1\u5349\u8bc6\u522b\u548c\u8ffd\u8e2a\n"
"    - \u7ed8\u5236\u5e76\u5c55\u793a\u51c6\u786e\u7387\u548c\u635f\u5931\u7387\u66f2\u7ebf\u56fe\n"
"    - \u5355\u6444\u50cf\u5934\u4e0b\u7684\u76ee\u6807\u8ffd\u8e2a\n"
"    - \u663e\u793a\u56fe\u50cf\u4e2d\u7684\u6bcf\u4e2a\u82b1\u6735\u7684\u79cd\u7c7b\u548c"
                        "\u7f6e\u4fe1\u5ea6\n"
"    \n"
"    \u9879\u76ee\u4f7f\u7528\u4e86\u4e00\u4e9b\u5f00\u6e90\u4ee3\u7801\uff0c\u5177\u4f53\u51fa\u5904\u5df2\u5728\u9879\u76ee\u4e2d\u6807\u660e\u3002\n"
"", None))
    # retranslateUi

