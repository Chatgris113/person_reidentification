from libs.camera import VideoCamera
import cv2
from libs.interactive_detection import Detections
from libs.argparser import build_argparser
from openvino.inference_engine import get_version
import configparser
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QMenuBar, QAction, QFileDialog, QDialog, QFormLayout, QLineEdit, QScrollArea, QDockWidget, QGridLayout
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, QtWidgets

# 讀取 config.ini
config = configparser.ConfigParser()
config.read("config.ini")
resize_width = int(config.get("CAMERA", "resize_width"))

# 全域變數
frame_id = 0

# 主視窗影像執行緒
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, flip_code):
        super().__init__()
        self.flip_code = flip_code
        self.camera = camera

    def run(self):
        frame_id = 0
        while True:
            frame_id += 1
            frame = self.camera.get_frame(self.flip_code)
            if frame is None:
                break
            self.frame_signal.emit(frame)

# 設定視窗
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")

        # create form layout
        form_layout = QFormLayout()

        # create async detection combo box
        self.async_detection_combo = QtWidgets.QComboBox()
        self.async_detection_combo.addItem(" ")
        self.async_detection_combo.addItem("True")
        self.async_detection_combo.addItem("False")
        form_layout.addRow("Async Detection", self.async_detection_combo)

        # create detection combo box
        self.detection_combo = QtWidgets.QComboBox()
        self.detection_combo.addItem(" ")
        self.detection_combo.addItem("True")
        self.detection_combo.addItem("False")
        form_layout.addRow("Detection", self.detection_combo)

        # create re-identification combo box
        self.reid_combo = QtWidgets.QComboBox()
        self.reid_combo.addItem(" ")
        self.reid_combo.addItem("True")
        self.reid_combo.addItem("False")
        form_layout.addRow("Re-Identification", self.reid_combo)

        # create show track combo box
        self.show_track_combo = QtWidgets.QComboBox()
        self.show_track_combo.addItem(" ")
        self.show_track_combo.addItem("True")
        self.show_track_combo.addItem("False")
        form_layout.addRow("Show Track", self.show_track_combo)

        # create save button
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        form_layout.addRow(save_button)

        # set layout
        self.setLayout(form_layout)

    def save_settings(self):
        # update main window settings
        self.parent().is_async = self.async_detection_combo.currentText() == "True"
        self.parent().is_det = self.detection_combo.currentText() == "True"
        self.parent().is_reid = self.reid_combo.currentText() == "True"
        self.parent().show_track = self.show_track_combo.currentText() == "True"

        # close dialog
        self.close()

# zoom視窗
class PersonDialog(QDialog):
    def __init__(self, parent=None, frame=None):
        super().__init__(parent)
        self.setWindowTitle("Zoom")
        self.frame = frame

        # create label for video frame
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setPixmap(QPixmap.fromImage(QImage(
            self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)))

        # create main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        self.setLayout(main_layout)

        # connect to parent's video thread
        self.parent().video_thread.frame_signal.connect(self.update_video)

    @ pyqtSlot(np.ndarray)
    def update_video(self, frame):
        # resize frame to fit label
        frame = cv2.resize(frame, (640, 480))
        # convert frame to pixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        # set pixmap to label
        self.video_label.setPixmap(pixmap)

# 主視窗
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_async = True
        self.is_det = True
        self.is_reid = True
        self.show_track = False
        self.flip_code = None

        self.setWindowTitle("ReID")  # 視窗標題

        self.video_thread = VideoThread(self.flip_code)
        self.video_thread.frame_signal.connect(self.update_video)
        self.video_thread.start()

        # create menu bar
        menu_bar = self.menuBar()

        # 設定選單
        settings_menu = menu_bar.addMenu('設定')
        settings_action = QAction('設定', self)
        settings_action.triggered.connect(self.show_settings_dialog)
        settings_menu.addAction(settings_action)
        # 說明選單
        explain_menu = menu_bar.addMenu('說明')
        # create status bar
        self.statusBar()
        # create video label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        # create main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)

        # create scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        # create widget for scroll area
        self.scroll_widget = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_widget)

        # create buttons layout
        self.buttons_layout = QVBoxLayout()
        self.buttons_layout.setSpacing(20)
        self.buttons = []
        buttons_widget = QWidget()
        buttons_widget.setLayout(self.buttons_layout)
        self.scroll_widget.setLayout(self.buttons_layout)
        buttons_dock = QDockWidget("偵測ID", self)
        buttons_dock.setWidget(self.scroll_area)
        self.addDockWidget(Qt.RightDockWidgetArea, buttons_dock)

       # set central widget
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.video_label)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    @ pyqtSlot(np.ndarray)
    def update_video(self, frame):
        if self.flip_code is not None:
            frame = cv2.flip(frame, self.flip_code)

        frame, detections_list = detections.person_detection(
            frame, self.is_async, self.is_det, self.is_reid, str(
                frame_id), self.show_track
        )
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

        # update buttons
        '''for button in self.buttons:
            button.setParent(None)
            self.buttons = []
        for detection in detections_list:
            button = QPushButton(f"ID: {detection['id']}")
            button.setFixedHeight(20)
            button.setContentsMargins(10, 10, 10, 10)
            self.buttons.append(button)
            self.buttons_layout.addWidget(button)   # 顯示於列表中
            button.clicked.connect(self.show_person_window)'''

    def closeEvent(self, event):
        self.video_thread.quit()
        self.video_thread.wait()

    def show_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def show_person_window(self):
        # get current frame
        frame = camera.frame
        # frame = self.video_thread.camera.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # create new dialog
        dialog = PersonDialog(parent=self, frame=frame)
        dialog.exec_()


if __name__ == "__main__":
    # arg parse
    args = build_argparser().parse_args()
    devices = [args.device, args.device_reidentification]
    camera = VideoCamera(args.input, resize_width, args.v4l)
    detections = Detections(camera.frame, devices, args.grid)
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
