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
from collections import deque
from logging import getLogger, basicConfig, DEBUG, INFO

logger = getLogger(__name__)

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

size_scale: float = 1.0

# 辨識執行緒
class detThread(QThread):
    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.camera = camera    # camera from main
        self.stopped = False

    def run(self) -> None:
        global detections_list  # 修改全域變數 detections_list
        global det_time_det
        global det_time_reid
        global det_fps

        while self.stopped is not True:
            frame = self.camera.get_frame(flip_code)
            if frame is None:
                break

            frame, detections_list = detections.person_detection(   # is_async = True
                frame, True, is_det, is_reid, str(frame_id), show_track
            )
            det_time_det, det_time_reid = detections.get_det_time()
            det_fps = detections.get_fps()

    def quit(self) -> None:
        self.stopped = True
        return super().quit()

# 主視窗畫面執行緒
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.camera = camera
        self.stopped = False

    def draw_stats(self, frame):
        # time
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
            if detection.get('track_points') is None:
                break
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
        while self.stopped is not True:
            frame_id += 1
            frame = self.camera.get_frame(flip_code)
            if frame is None:
                break

            # 畫狀態
            self.draw_stats(frame)
            # 畫框框
            self.draw_box(frame)
            # 畫軌跡
            if is_reid and show_track:
                self.draw_track_points(frame)

            self.frame_signal.emit(frame)

    def quit(self) -> None:
        self.stopped = True
        return super().quit()

# zoom 執行緒
# 備註: 由於使用 track_points 做穩定，目前只能用於 is_reid = True
class ZoomVideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent: QDialog) -> None:
        super().__init__(parent)
        self.camera = camera
        self.stopped = False

        self.avg_num = 20
        self.bbox_list = deque([], maxlen=self.avg_num)    # bbox list(deque)
        
        # 取得 id
        title = parent.windowTitle()
        index = title.find('Zoom ID: ')
        if index > -1:
            self.id = int(title[index + 9:])

    def avg(self):
        # bbox = (xmin, ymin, xmax, ymax)
        xmid = 0
        x_list = [] # x mid
        y_list = [] # y min

        for bbox in self.bbox_list:
            xmid = int((bbox[0] + bbox[2]) / 2)
            x_list.append(xmid)
            y_list.append(bbox[1])
        
        xmid = int(sum(x_list) / len(x_list))
        ymin = int(sum(y_list) / len(y_list))

        return xmid, ymin
    
    def crop(self, frame, x_mid, y_min):
        x = x_mid - 250
        y = y_min - 10
        if x < 0: x = 0
        if y < 0: y = 0

        w = 500
        h = 300

        frame = frame[y:y+h, x:x+w]

        return frame

    def run(self) -> None:
        while self.stopped is not True:
            frame = self.camera.get_frame(flip_code)
            if frame is None:
                break
            # code here
            for detection in detections_list:
                if detection['id'] is not self.id:
                    continue
                bbox = detection['bbox']
                self.bbox_list.append(bbox)
                if len(self.bbox_list) >= self.avg_num:     # 需要至少 n 個才執行平均
                    x_avg, y_avg = self.avg()
                    frame = self.crop(frame, x_avg, y_avg)

            self.frame_signal.emit(frame)
    
    def quit(self) -> None:
        self.stopped = True
        return super().quit()

# 設定視窗
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")

        # create form layout
        form_layout = QFormLayout()

        # create detection combo box
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("None")
        self.mode_combo.addItem("Detection")
        self.mode_combo.addItem("Re-Identification")
        form_layout.addRow("Mode Select", self.mode_combo)

        # create show track combo box
        self.show_track_combo = QtWidgets.QComboBox()
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
        if self.mode_combo.currentText() == "None":
            is_det = False
            is_reid = False
        elif self.mode_combo.currentText() == "Detection":
            is_det = True
            is_reid = False
        elif self.mode_combo.currentText() == "Re-Identification":
            is_det = True
            is_reid = True
        show_track = self.show_track_combo.currentText() == "True"

        # close dialog
        self.close()

# 縮放設定視窗
class ScaleSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("縮放設定")

        # create form layout
        form_layout = QFormLayout()

        self.Scale_Slider = QtWidgets.QSlider(self)
        #self.Scale_Slider.setGeometry(QtCore.QRect(100, 150, 160, 22))
        self.Scale_Slider.setMinimum(10)
        self.Scale_Slider.setMaximum(50)
        self.Scale_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Scale_Slider.setObjectName("scale_slider")
        form_layout.addRow("Scale Slider", self.Scale_Slider)
        '''# create detection combo box
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("None")
        self.mode_combo.addItem("Detection")
        self.mode_combo.addItem("Re-Identification")
        form_layout.addRow("Mode Select", self.mode_combo)

        # create show track combo box
        self.show_track_combo = QtWidgets.QComboBox()
        self.show_track_combo.addItem("True")
        self.show_track_combo.addItem("False")
        form_layout.addRow("Show Track", self.show_track_combo)'''

        # create save button
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        form_layout.addRow(save_button)

        # set layout
        self.setLayout(form_layout)

    def save_settings(self):
        print(self.Scale_Slider.value())

        # close dialog
        self.close()

# zoom視窗
class PersonDialog(QDialog):
    def __init__(self, parent=None, Title="Zoom"):
        super().__init__(parent)
        self.setWindowTitle(Title)

        # create label for video frame
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)

        # create main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        self.setLayout(main_layout)

        # connect to zoom video thread
        self.zoom_thread = ZoomVideoThread(self)
        self.zoom_thread.frame_signal.connect(self.update_video)
        self.zoom_thread.start()

    @ pyqtSlot(np.ndarray)
    def update_video(self, frame: cv2.Mat):
        # convert frame to pixmap
        height, width, channel = frame.shape
        frame = cv2.resize(frame, (width, height))
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        # set pixmap to label
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.zoom_thread.quit()
        self.zoom_thread.wait()
        event.accept()

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
        # scale_settings_action = QAction('縮放設定', self)
        # scale_settings_action.triggered.connect(self.show_scale_settings_dialog)
        # settings_menu.addAction(scale_settings_action)
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

        # zoom dialog
        self.dialogs: List[QDialog] = []

        # Test: buttons
        for id in range(1, 50):
            button = QPushButton(f"ID: {id}")
            button.setObjectName(f"btn_{id}")
            button.setFixedHeight(20)
            button.setContentsMargins(10, 10, 10, 10)
            self.buttons.append(button)
            self.buttons_layout.addWidget(button)

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
                id = detection['id']
                button = QPushButton(f"ID: {id}")
                button.setObjectName(f"btn_{id}")
                button.setFixedHeight(20)
                button.setContentsMargins(10, 10, 10, 10)
                self.buttons.append(button)
                self.buttons_layout.addWidget(button)   # 顯示於列表中
                button.clicked.connect(self.show_person_window)

        '''
        for button in self.buttons:
            det_exist: bool = False
            for detection in detections_list:
                if f"ID: {detection['id']}" == button.text():
                    det_exist = True
            if det_exist == False:
                button.hide()
        '''

    def closeEvent(self, event):
        self.det_thread.quit()
        self.det_thread.wait()
        self.video_thread.quit()
        self.video_thread.wait()
        event.accept()

    def show_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def show_scale_settings_dialog(self):
        dialog = ScaleSettingsDialog(self)
        dialog.exec_()

    def show_person_window(self):
        sender: QPushButton = self.sender()
        title = f"Zoom {sender.text()}"
        # 從 dialogs 移除已關掉的 dialog
        for dialog in self.dialogs:
            if dialog.windowTitle() == title: # 若重複則先關閉
                dialog.close()
            if dialog.isVisible() == False:
                self.dialogs.remove(dialog)
        # create new dialog
        dialog = PersonDialog(parent=self, Title = title)
        dialog.show()
        self.dialogs.append(dialog)


if __name__ == "__main__":
    level = INFO
    basicConfig(
        filename="app.log",
        filemode="w",
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s",
    )

    # arg parse
    args = build_argparser().parse_args()
    devices = [args.device, args.device_reidentification]
    camera = VideoCamera(args.input, resize_width, args.v4l)
    detections = Detections(camera.frame, devices, args.grid)
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
