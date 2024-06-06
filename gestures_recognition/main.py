import os
import cv2
import copy
import time
from itertools import chain
from typing import List, Tuple
import numpy as np
import mediapipe as mp

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
        self.MODES_NUMBER = 3
        self.mode = 0
        self._is_data_recording = False
        self._save_single_record = False
        self.image_label = None
        self._image_label_buffer = []

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
                    for hand_landmarks, _ in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        landmarks_list = self._calc_landmarks_list(
                            image_copy, hand_landmarks
                        )

                        bounding_box = self._cals_bounding_box(landmarks_list)

                        image_copy = self.draw_bounding_box(
                            self.show_bounding_boxes, image_copy, bounding_box
                        )

                        if self._save_single_record:
                            self._save_image_part(image, bounding_box)
                            self._save_single_record = False
                        
                        if self.mode == 2 and self._is_data_recording:
                            self._save_image_part(image, bounding_box)
                            
                current_process_time = time.time()
                fps = int(1 / (current_process_time - prev_process_time))
                prev_process_time = current_process_time
                
                cv2.putText(image_copy, f"FPS: {fps}", (20, 30), FONT, 0.7, COLOR_WHITE)
                cv2.putText(image_copy, f'Mode: {self.mode} ("M" to change)', (20, 55), FONT, 0.7, COLOR_WHITE)
                
                if self.mode == 1 or self.mode == 2: 
                    cv2.putText(image_copy, f"Label: {self.image_label}", (20, 115), FONT, 0.7, COLOR_WHITE)
                if self.mode == 2:
                    cv2.putText(image_copy, f"Data collection mode", (20, 90), FONT, 0.7, COLOR_WHITE)
                    guide_text = f'Press "S" to START recording' if not self._is_data_recording else f'Press "S" to STOP recording'
                    cv2.putText(image_copy, guide_text, (20, 140), FONT, 0.7, COLOR_WHITE)
                    cv2.putText(image_copy, 'Press "N" to save a single record', (20, 165), FONT, 0.7, COLOR_WHITE)
                        
                
            
                cv2.imshow("MediaPipe Hands", image_copy)

            cap.release()
            cv2.destroyAllWindows()
            
    def _save_image_part(self, image, bounding_box):
        save_part = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
        save_part = cv2.cvtColor(save_part, cv2.COLOR_BGR2RGB)
        save_part = cv2.resize(save_part, (224, 224))
        
        if self.image_label is None:
            raise ValueError("Label is not defined!")
        
        data_folder = f"data/{self.image_label}"
        os.makedirs(f"data/{self.image_label}", exist_ok=True)
        
        file_counter = 0
        while True:
            filename = f"image_{file_counter}.jpg"
            filepath = os.path.join(data_folder, filename)

            if not os.path.exists(filepath):
                break  # Unique filename found
            file_counter += 1 
                
        cv2.imwrite(filepath, save_part)

    def _change_recording_state(self, key):
        if key == 115: # "s" -> start/stop recording
            self._is_data_recording = not self._is_data_recording
        if key == 110: # "n" -> make a single record
            self._save_single_record = True

    def _select_image_label(self, key):
        if 97 <= key <= 122:
            self._image_label_buffer.append(chr(key))
        elif key == 8 and self._image_label_buffer: # key == 8 - backspace
            self._image_label_buffer.pop()
            
        if self._image_label_buffer:
            self.image_label = "".join(self._image_label_buffer)
        else:
            self.image_label = None

    def _select_mode(self, key):
        if key == 109: # 'm' - change mode
            self.mode = (self.mode + 1) % self.MODES_NUMBER

    def process_settings(self, key):
        if not self._is_data_recording:
            self._select_mode(key)
            
        if self.mode == 1:
            self._select_image_label(key)
        
        if self.mode == 2:
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
        x = int(x-50)
        y = int(y-50)
        w = int(w+100)
        h = int(h+100)
        
        return [x, y, x + w, y + h]


def main():
    landmarks_detector = MediaPipeHandLandmarks(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    landmarks_detector.get_landmarks_from_stream(0)


if __name__ == "__main__":
    main()