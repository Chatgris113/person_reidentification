from libs.camera import VideoCamera
import cv2
from libs.interactive_detection import Detections
from libs.argparser import build_argparser
from openvino.inference_engine import get_version
import configparser
from PyQt5.QtCore import QObject, pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QMenuBar, QAction, QFileDialog, QDialog, QFormLayout, QLineEdit, QScrollArea, QDockWidget, QGridLayout
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, QtWidgets
import sys
from typing import List

# 讀取 config.ini
config = configparser.ConfigParser()
config.read("config.ini")
resize_width = int(config.get("CAMERA", "resize_width"))
prob_thld_person = eval(config.get("DETECTION", "prob_thld_person"))

# 全域變數
frame_id: int = 0
detections_list = []
is_det: bool = True
is_reid: bool = True
show_track: bool = True
flip_code: int = 1

det_time_det = 0
det_time_reid = 0
det_fps = "FPS: ??"

# 辨識執行緒
class detThread(QThread):
    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.camera = camera    # camera from main

    def run(self) -> None:
        global detections_list  # 修改全域變數 detections_list
        global det_time_det
        global det_time_reid
        global det_fps

        while True:
            frame = self.camera.get_frame(flip_code)
            frame, detections_list = detections.person_detection(   # is_async = True
                frame, True, is_det, is_reid, str(frame_id), show_track
            )
            det_time_det, det_time_reid = detections.get_det_time()
            det_fps = detections.get_fps()

# 主視窗畫面執行緒
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.camera = camera

    def draw_stats(self, frame):
        # texts
        det_time = det_time_det + det_time_reid
        inf_time_message = (
            f"Total Inference time: {det_time * 1000:.3f} ms for async mode"
        )
        cv2.putText(
            frame,
            inf_time_message,
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (10, 10, 200),
            1,
        )
        det_time_txt = None
        if is_det or is_reid:
            det_time_txt = f"det:{det_time_det * 1000:.3f} ms "
        if is_reid:
            det_time_txt = det_time_txt + \
                f"reid:{det_time_reid * 1000:.3f} ms"
        if det_time_txt is not None:
            inf_time_message = (
                f"@Detection prob:{prob_thld_person} time: {det_time_txt}"
            )
            cv2.putText(
                frame,
                inf_time_message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (10, 10, 200),
                1,
            )
        # det FPS, det person count
        cv2.rectangle(
            frame, 
            (frame.shape[1] - 75, 0), 
            (frame.shape[1], 34), 
            (255, 255, 255), 
            -1
        )
        cv2.putText(
            frame,
            f"DET {det_fps}",
            (frame.shape[1] - 75 + 3, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )
        person_counter = len(detections_list)
        cv2.putText(
            frame,
            f"DET: {person_counter}",
            (frame.shape[1] - 75 + 3, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )
        return frame

    def draw_box(self, frame):
        for detection in detections_list:
            result = f"{detection['id']} {detection['confidence']}%"
            size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            xmin, ymin, xmax, ymax = detection['bbox']
            green = (0, 255, 0)
            xtext = xmin + size[0][0] + 20
            cv2.rectangle(
                frame, (xmin, ymin - 22), (xtext, ymin), green, -1,
            )
            cv2.rectangle(
                frame, (xmin, ymin - 22), (xtext, ymin), green,
            )
            cv2.rectangle(
                frame, (xmin, ymin), (xmax, ymax), green, 1,
            )
            cv2.putText(
                frame,
                result,
                (xmin + 3, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
        return frame
    
    def draw_track_points(self, frame):
        for detection in detections_list:
            tp = np.array(detection['track_points'])
            track_points = tp[~np.isnan(tp).any(axis=1)].astype(int)
            if len(track_points) > 2:
                track_points = np.array(track_points)
                green = (0, 255, 0)
                cv2.polylines(
                    frame, [track_points], isClosed=False, color=green, thickness=2,
                )
        return frame

    def run(self):
        global frame_id

        frame_id = 0
        while True:
            frame_id += 1
            frame = self.camera.get_frame(flip_code)
            if frame is None:
                break

            # 畫狀態
            self.draw_stats(frame)
            # 畫框框
            self.draw_box(frame)
            # 畫軌跡
            if is_reid:
                self.draw_track_points(frame)

            self.frame_signal.emit(frame)

# 設定視窗
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")

        # create form layout
        form_layout = QFormLayout()

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
        global is_det
        global is_reid
        global show_track

        # update main window settings
        is_det = self.detection_combo.currentText() == "True"
        is_reid = self.reid_combo.currentText() == "True"
        show_track = self.show_track_combo.currentText() == "True"

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

        self.setWindowTitle("ReID")  # 視窗標題

        self.video_thread = VideoThread(self)
        self.video_thread.frame_signal.connect(self.update_video)
        self.video_thread.start()

        self.det_thread = detThread(self)
        self.det_thread.start()

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
        self.buttons: List[QPushButton] = []
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
    def update_video(self, frame):  # frame from VideoThread
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

        # update buttons
        for detection in detections_list:
            button_exist: bool = False
            for button in self.buttons:
                if button.text() == f"ID: {detection['id']}":
                    button.show()
                    button_exist = True
            if button_exist == False:
                button = QPushButton(f"ID: {detection['id']}")
                button.setFixedHeight(20)
                button.setContentsMargins(10, 10, 10, 10)
                self.buttons.append(button)
                self.buttons_layout.addWidget(button)   # 顯示於列表中
                button.clicked.connect(self.show_person_window)

        for button in self.buttons:
            det_exist: bool = False
            for detection in detections_list:
                if f"ID: {detection['id']}" == button.text():
                    det_exist = True
            if det_exist == False:
                button.hide()


    def closeEvent(self, event):
        self.det_thread.quit()
        self.det_thread.wait()
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
    sys.exit(app.exec_())
