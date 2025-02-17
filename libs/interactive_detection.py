from logging import getLogger
import cv2
import numpy as np
from timeit import default_timer as timer
from libs.tracker import Tracker
import libs.detectors as detectors
import configparser

logger = getLogger(__name__)

config = configparser.ConfigParser()
config.read("config.ini")

# probability threshold to detect persons
prob_thld_person = eval(config.get("DETECTION", "prob_thld_person"))

# basic colors
green = eval(config.get("COLORS", "green"))
skyblue = eval(config.get("COLORS", "skyblue"))

# OpenVINO Models
model_path = config.get("MODELS", "model_path")
model_det = config.get("MODELS", "model_det")
model_reid = config.get("MODELS", "model_reid")


class Detectors:
    def __init__(self, devices):
        self.device_det, self.device_reid = devices
        self._define_models()
        self._load_detectors()

    def _define_models(self):
        # person detection
        fp_path = "FP16-INT8" if self.device_det == "CPU" else "FP16"
        self.model_det = f"{model_path}/{model_det}/{fp_path}/{model_det}.xml"
        # person reIdentification
        fp_path = "FP32" if self.device_reid == "CPU" else "FP16"
        self.model_reid = f"{model_path}/{model_reid}/{fp_path}/{model_reid}.xml"

    def _load_detectors(self):
        # person_detection
        self.person_detector = detectors.PersonDetection(
            self.device_det, self.model_det
        )
        # person re-identification
        self.person_id_detector = detectors.PersonReIdentification(
            self.device_reid, self.model_reid
        )


class Detections(Detectors):
    def __init__(self, frame, devices, grid):
        super().__init__(devices)

        # initialize Calculate FPS
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()
        self.prev_frame = frame
        # create tracker instance
        self.tracker = Tracker(self.person_id_detector, frame, grid)
        # inference time
        self.det_time_det = 0
        self.det_time_reid = 0

    def _calc_fps(self):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps += 1

        if self.accum_time > 1:
            self.accum_time += -1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

    def draw_bbox(self, frame, box, result, color):
        xmin, ymin, xmax, ymax = box
        size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
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

    def draw_perf_stats(
        self, det_time, det_time_txt, frame, frame_id, is_async, person_counter=None
    ):

        # Draw FPS on top right corner
        self._calc_fps()
        cv2.rectangle(
            frame, 
            (frame.shape[1] - 50, 0), 
            (frame.shape[1], 17), 
            (255, 255, 255), 
            -1
        )
        cv2.putText(
            frame,
            self.fps,
            (frame.shape[1] - 50 + 3, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )
        # Draw Real-Time Person Counter on top right corner
        if person_counter is not None:
            cv2.rectangle(
                frame,
                (frame.shape[1] - 50, 17),
                (frame.shape[1], 34),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                frame,
                f"DET: {person_counter}",
                (frame.shape[1] - 50 + 3, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
            )

        # Draw Frame number at bottom corner
        cv2.rectangle(
            frame,
            (frame.shape[1] - 50, frame.shape[0] - 20),
            (frame.shape[1], frame.shape[0]),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame,
            frame_id.zfill(5),
            (frame.shape[1] - 40, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )

        # Draw performance stats
        if is_async:
            inf_time_message = (
                f"Total Inference time: {det_time * 1000:.3f} ms for async mode"
            )
        else:
            inf_time_message = (
                f"Total Inference time: {det_time * 1000:.3f} ms for sync mode"
            )
        cv2.putText(
            frame,
            inf_time_message,
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 10, 10),
            1,
        )
        if det_time_txt:
            inf_time_message = (
                f"@Detection prob:{prob_thld_person} time: {det_time_txt}"
            )
            cv2.putText(
                frame,
                inf_time_message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 10, 10),
                1,
            )
        return frame

    def get_person_frames(self, persons, frame):

        frame_h, frame_w = frame.shape[:2]
        person_frames = []
        boxes = []

        for person in persons[0][0]:
            box = person[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            xmin, ymin, xmax, ymax = box.astype("int")
            person_frame = frame[ymin:ymax, xmin:xmax]
            person_h, person_w = person_frame.shape[:2]
            # Resizing person_frame will be failed when witdh or height of the person_fame is 0
            # ex. (243, 0, 3)
            if person_h != 0 and person_w != 0:
                boxes.append((xmin, ymin, xmax, ymax))
                person_frames.append(person_frame)

        return person_frames, boxes

    def person_detection(self, frame, is_async, is_det, is_reid, frame_id, show_track):
        # Give the frame_id with tracker instance
        self.tracker.frame_id = frame_id
        self.tracker.show_track = show_track

        # init params
        det_time = 0
        self.det_time_det = 0
        self.det_time_reid = 0
        persons = None
        person_frames = None
        boxes = None
        person_counter = 0
        person_info = None

        # just return frame when person detection and person reidentification are False
        if not is_det and not is_reid:
            self.prev_frame = self.draw_perf_stats(
                det_time, "Video capture mode", frame, frame_id, is_async
            )
            return self.prev_frame, []

        if is_async:
            prev_frame = frame.copy()
        else:
            self.prev_frame = frame.copy()

        if is_det or is_reid:

            # ----------- Person Detection ---------- #
            inf_start = timer()
            self.person_detector.infer(self.prev_frame, frame, is_async)
            persons = self.person_detector.get_results(
                is_async, prob_thld_person)
            inf_end = timer()
            self.det_time_det = inf_end - inf_start
            det_time_txt = f"person det:{self.det_time_det * 1000:.3f} ms "

            if persons is None:
                return self.prev_frame, []

            person_frames, boxes = self.get_person_frames(
                persons, self.prev_frame)
            person_counter = len(person_frames)
            confidence_list = []

            if is_det and person_frames:
                # ----------- Draw result into the frame ---------- #
                for det_id, person_frame in enumerate(person_frames):
                    confidence = round(persons[0][0][det_id][2] * 100, 1)
                    confidence_list.append(confidence)
                    result = f"{det_id} {confidence}%"
                    # draw bounding box per each person into the frame
                    self.prev_frame = self.draw_bbox(
                        self.prev_frame, boxes[det_id], result, green
                    )

            # ----------- Person ReIdentification ---------- #
            if is_reid:
                inf_start = timer()
                self.prev_frame, person_info = self.tracker.person_reidentification(
                    self.prev_frame, person_frames, boxes
                )
                inf_end = timer()
                self.det_time_reid = inf_end - inf_start
                det_time_txt = det_time_txt + \
                    f"reid:{self.det_time_reid * 1000:.3f} ms"

            if person_frames is None:
                det_time_txt = "No persons detected"

        det_time = self.det_time_det + self.det_time_reid
        frame = self.draw_perf_stats(
            det_time,
            det_time_txt,
            self.prev_frame,
            frame_id,
            is_async,
            person_counter=str(person_counter),
        )

        if is_async:
            self.prev_frame = prev_frame

        # 可能在 det mode
        if person_info is None:
            person_info = []
            for det_id, person_frame in enumerate(person_frames):
                bbox = boxes[det_id]
                confidence = confidence_list[det_id]
                person_dict = {
                    "id": det_id,
                    "bbox": bbox,
                    "frame": person_frame,
                    "confidence": confidence
                }
                person_info.append(person_dict)

        return frame, person_info
    
    def get_det_time(self):
        return self.det_time_det, self.det_time_reid

    def get_fps(self):
        return self.fps