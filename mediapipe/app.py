import cv2
import copy
import time
from itertools import chain
from typing import List, Tuple
import numpy as np
import mediapipe as mp
from utils import csvoperations
from model.gesture_classifier import GestureClassifier

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class MediaPipeHandLandmarks:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        show_bounding_boxes: bool = True,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.show_bounding_boxes = show_bounding_boxes
        self.MODES_NUMBER = 2
        self.mode = 0
        self.gesture_id = None
        self._gesture_id_buffer = []
        self._is_data_recording = False

    def get_landmarks_from_stream(self, camera_index: int = 0) -> None:
        cap = cv2.VideoCapture(camera_index)

        # Frame resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        prev_process_time = 0
        # Mode defines current behavior of the app
        # mode = 0 - regular hand tracking
        # mode = 1 - hand landmarks data collection

        FONT = cv2.FONT_HERSHEY_COMPLEX
        COLOR_WHITE = (255, 255, 255)
        
        gestures_labels = csvoperations.read_labels("model/data/gestures_labels.csv")
        gesture_classifier = GestureClassifier()

        with mp_hands.Hands(
            model_complexity=self.model_complexity,
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as hands:
            while cap.isOpened():
                key = cv2.waitKey(5)
                if key == 27:
                    break
                
                self.process_settings(key)
                
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break

                image_copy = copy.deepcopy(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        landmarks_list = self._calc_landmarks_list(
                            image_copy, hand_landmarks
                        )

                        bounding_box = self._cals_bounding_box(landmarks_list)

                        normalized_landmarks_list = self._normalize_landmarks_list(landmarks_list)
                        
                        hand_gesture_id = gesture_classifier(normalized_landmarks_list)
                        
                        image_copy = self.draw_bounding_box(
                            self.show_bounding_boxes, image_copy, bounding_box
                        )

                        image_copy = self.draw_landmarks(image_copy, landmarks_list)

                        image_copy = self.draw_text_info(
                            image_copy, handedness, bounding_box, gestures_labels[hand_gesture_id]
                        )
                        
                        if self.mode == 1 and self._is_data_recording:
                            if (normalized_landmarks_list is not None) and (self.gesture_id is not None):
                                csvoperations.save_data(self.gesture_id, normalized_landmarks_list, "model/data/landmarks_data.csv")

                current_process_time = time.time()
                fps = int(1 / (current_process_time - prev_process_time))
                prev_process_time = current_process_time
                
                cv2.putText(image_copy, f"FPS: {fps}", (20, 30), FONT, 0.7, COLOR_WHITE)
                cv2.putText(image_copy, f'Mode: {self.mode} ("M" to change)', (20, 55), FONT, 0.7, COLOR_WHITE)
                
                if self.mode == 1:
                    cv2.putText(image_copy, f"Data collection mode", (20, 90), FONT, 0.7, COLOR_WHITE)
                    cv2.putText(image_copy, f"Gesture ID: {self.gesture_id}", (20, 115), FONT, 0.7, COLOR_WHITE)
                    guide_text = f'Press "S" to START recording' if not self._is_data_recording else f'Press "S" to STOP recording'
                    cv2.putText(image_copy, guide_text, (20, 140), FONT, 0.7, COLOR_WHITE)
                        
                
            
                cv2.imshow("MediaPipe Hands", image_copy)

            cap.release()
            cv2.destroyAllWindows()
            
    def _normalize_landmarks_list(self, landmarks_list):
        tmp_landmarks = None
        if landmarks_list:
            tmp_landmarks = copy.deepcopy(landmarks_list)
            base_x, base_y = tmp_landmarks[0][0], tmp_landmarks[0][1]
            for i in range(len(tmp_landmarks)):
                tmp_landmarks[i][0] = tmp_landmarks[i][0] - base_x
                tmp_landmarks[i][1] = tmp_landmarks[i][1] - base_y

            tmp_landmarks = list(chain.from_iterable(tmp_landmarks))
            
            max_value = max(list(map(abs, tmp_landmarks)))
            
            tmp_landmarks = [value / max_value for value in tmp_landmarks]
            
        return tmp_landmarks

    def _select_mode(self, key):
        if key == 109: # 'm' - change mode
            self.mode = (self.mode + 1) % self.MODES_NUMBER
            
    def _change_recording_state(self, key):
        if key == 115:
            self._is_data_recording = not self._is_data_recording

    def _select_gesture_id(self, key):
        if 48 <= key <= 57: # numbers from 0 to 9
            self._gesture_id_buffer.append(key-48)
        elif key == 8 and self._gesture_id_buffer: # key == 8 - backspace
            self._gesture_id_buffer.pop()
        
        if self._gesture_id_buffer:
            self.gesture_id = int("".join(map(str, self._gesture_id_buffer)))
        else:
            self.gesture_id = None

    def process_settings(self, key):
        if not self._is_data_recording:
            self._select_mode(key)
            self._select_gesture_id(key)
        
        if self.mode == 1:
            self._change_recording_state(key)

    def draw_bounding_box(self, show_box, image, bounding_box):
        if show_box:
            cv2.rectangle(
                img=image,
                pt1=(bounding_box[0], bounding_box[1]),
                pt2=(bounding_box[2], bounding_box[3]),
                color=(0, 0, 0),
                thickness=1,
            )
        return image

    def _calc_landmarks_list(self, image, landmarks):
        img_height, img_width = image.shape[0], image.shape[1]

        landmark_points = []

        for landmark in landmarks.landmark:
            x = min(int(landmark.x * img_width), img_width - 1)
            y = min(int(landmark.y * img_height), img_height - 1)

            landmark_points.append([x, y])

        return landmark_points

    def _cals_bounding_box(self, landmarks_list):
        landmark_array = np.array(landmarks_list)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y - 5, x + w, y + h]

    def _connect_points_with_lines(
        self,
        image,
        points: List[Tuple[int]],
        inner_color: Tuple[int] = (255, 255, 255),
        outer_color: Tuple[int] = (0, 0, 0),
        inner_width: int = 2,
        outer_width: int = 6,
    ) -> None:
        if len(points) > 0:
            for i in range(len(points) - 1):
                cv2.line(
                    img=image,
                    pt1=tuple(points[i]),
                    pt2=tuple(points[i + 1]),
                    color=outer_color,
                    thickness=outer_width,
                )
                cv2.line(
                    img=image,
                    pt1=tuple(points[i]),
                    pt2=tuple(points[i + 1]),
                    color=inner_color,
                    thickness=inner_width,
                )

    def draw_landmarks(
        self, image: np.ndarray, landmarks_list: List[List[int]]
    ) -> np.ndarray:
        if len(landmarks_list) > 0:
            palm_indicies = [1, 0, 5, 9, 13, 17, 0]
            palm = [landmarks_list[i] for i in palm_indicies]
            fingers = [landmarks_list[i : i + 4] for i in range(1, len(landmarks_list), 4)]

            for finger in fingers:
                self._connect_points_with_lines(image, finger)

            self._connect_points_with_lines(image, palm)

            # Key Points
            for index, landmark in enumerate(landmarks_list):
                fingertips_indices = (4, 8, 12, 16, 20)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                if index in fingertips_indices:
                    cv2.circle(image, (landmark[0], landmark[1]), 4, (0, 0, 255), -1)
                else:
                    cv2.circle(image, (landmark[0], landmark[1]), 4, (255, 255, 255), -1)
                    
        return image

    def draw_text_info(
        self,
        image,
        handedness,
        bounding_box,
        hand_gesture_text,
        text_box_height: int = 22,
        text_box_color: Tuple[int] = (0, 0, 0),
        text_color: Tuple[int] = (255, 255, 255),
    ):
        cv2.rectangle(
            img=image,
            pt1=(bounding_box[0], bounding_box[1]),
            pt2=(bounding_box[2], bounding_box[1] - text_box_height),
            color=text_box_color,
            thickness=-1,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        handedness_text = handedness.classification[0].label
        text_info = "" + handedness_text
        if hand_gesture_text != "":
            text_info += ": " + hand_gesture_text
        cv2.putText(
            img=image,
            text=text_info,
            org=(bounding_box[0] + 5, bounding_box[1] - 5),
            fontFace=font,
            fontScale=0.6,
            color=text_color,
        )

        return image


def main():
    landmarks_detector = MediaPipeHandLandmarks(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    landmarks_detector.get_landmarks_from_stream(0)


if __name__ == "__main__":
    main()
