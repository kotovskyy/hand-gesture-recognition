import os
import cv2
import copy
import time
from typing import List, Tuple
import numpy as np
import mediapipe as mp
from model.gesture_classifier import GestureClassifier
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

LABELS = ["Closed", "Okay", "Open", "Left", "Right"]


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
        self._mirror_image = False
        self._last_image_index = None
        self.last_gesture_id = None
        self.last_gesture_time = time.time()

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



        gesture_classifier = GestureClassifier("models/sign_net_v6.tflite")

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

                if self._mirror_image:
                    image = cv2.flip(image, 1)
                
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

                        try:
                            hand_part = self._prep_image_part(image_copy, bounding_box)
                            hand_gesture_id = gesture_classifier(hand_part)
                            

                            if time.time() - self.last_gesture_time > 1 and hand_gesture_id == self.last_gesture_id:
                                if hand_gesture_id == 0:
                                    pyautogui.press("backspace")
                                elif hand_gesture_id == 1:
                                    pyautogui.press("enter")
                                elif hand_gesture_id == 2:
                                    pyautogui.press("space")
                                elif hand_gesture_id == 3:
                                    pyautogui.press("left")
                                elif hand_gesture_id == 4:
                                    pyautogui.press("right")

                                self.last_gesture_time = time.time()
                            else:
                                self.last_gesture_id = hand_gesture_id


                        except Exception as e:
                            pass

                        image_copy = self.draw_bounding_box(
                            self.show_bounding_boxes, image_copy, bounding_box
                        )

                        image_copy = self.draw_text_info(
                            image_copy, bounding_box, LABELS[hand_gesture_id]
                        )

                        if self._save_single_record:
                            try:
                                self._save_image_part(image, bounding_box)
                            except Exception as e:
                                pass
                            finally:
                                self._save_single_record = False

                        if self.mode == 2 and not self._is_data_recording:
                            self._last_image_index = None
                        
                        if self.mode == 2 and self._is_data_recording:
                            try:
                                self._save_image_part(image, bounding_box)
                            except Exception as e:
                                pass
                else:
                    self.last_gesture_id = None
                    self.last_gesture_time = time.time()

                current_process_time = time.time()
                fps = int(1 / (current_process_time - prev_process_time))
                prev_process_time = current_process_time

                cv2.putText(image_copy, f"FPS: {fps}", (20, 30), FONT, 0.7, COLOR_WHITE)
                cv2.putText(
                    image_copy,
                    f'Mode: {self.mode} ("M" to change)',
                    (20, 55),
                    FONT,
                    0.7,
                    COLOR_WHITE,
                )

                if self.mode == 1 or self.mode == 2:
                    cv2.putText(
                        image_copy,
                        f"Label: {self.image_label}",
                        (20, 115),
                        FONT,
                        0.7,
                        COLOR_WHITE,
                    )
                if self.mode == 2:
                    cv2.putText(
                        image_copy,
                        f"Data collection mode",
                        (20, 90),
                        FONT,
                        0.7,
                        COLOR_WHITE,
                    )
                    guide_text = (
                        f'Press "S" to START recording'
                        if not self._is_data_recording
                        else f'Press "S" to STOP recording'
                    )
                    cv2.putText(
                        image_copy, guide_text, (20, 140), FONT, 0.7, COLOR_WHITE
                    )
                    cv2.putText(
                        image_copy,
                        'Press "N" to save a single record',
                        (20, 165),
                        FONT,
                        0.7,
                        COLOR_WHITE,
                    )

                cv2.imshow("MediaPipe Hands", image_copy)

            cap.release()
            cv2.destroyAllWindows()

    def _prep_image_part(self, image, bounding_box):
        save_part = image[
            bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]
        ]
        save_part = cv2.cvtColor(save_part, cv2.COLOR_BGR2RGB)
        save_part = cv2.resize(save_part, (224, 224))

        return save_part

    def _save_image_part(self, image, bounding_box):
        save_part = self._prep_image_part(image, bounding_box)

        if self.image_label is None:
            raise ValueError("Label is not defined!")

        data_folder = f"data/{self.image_label}"
        os.makedirs(f"data/{self.image_label}", exist_ok=True)

        if self._last_image_index is None: 
            file_counter = 0
            while True:
                filename = f"image_{file_counter}.jpg"
                filepath = os.path.join(data_folder, filename)

                if not os.path.exists(filepath):
                    self._last_image_index = file_counter
                    break  # Unique filename found
                file_counter += 1
        else:
            filename = f"image_{self._last_image_index}.jpg"
            filepath = os.path.join(data_folder, filename)
            self._last_image_index += 1

        print(self._last_image_index)
        cv2.imwrite(filepath, save_part)

    def _change_recording_state(self, key):
        if key == 115:  # "s" -> start/stop recording
            self._is_data_recording = not self._is_data_recording
        if key == 110:  # "n" -> make a single record
            self._save_single_record = True

    def _select_image_label(self, key):
        if 97 <= key <= 122 or 65 <= key <= 90:
            self._image_label_buffer.append(chr(key))
        elif key == 8 and self._image_label_buffer:  # key == 8 - backspace
            self._image_label_buffer.pop()

        if self._image_label_buffer:
            self.image_label = "".join(self._image_label_buffer)
        else:
            self.image_label = None

    def _select_mode(self, key):
        if key == 109:  # 'm' - change mode
            self.mode = (self.mode + 1) % self.MODES_NUMBER

    def process_settings(self, key):
        "Process the app settings"
        if not self._is_data_recording:
            self._select_mode(key)

        if self.mode == 1:
            self._select_image_label(key)

        if self.mode == 2:
            self._change_recording_state(key)
            
        if key == 102: # "f" - flip image
            self._mirror_image = not self._mirror_image
            

    def draw_bounding_box(self, show_box, image, bounding_box):
        "Draw bounding box on the image"
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
        "Calculates the list of landmarks from the given landmarks object."
        img_height, img_width = image.shape[0], image.shape[1]

        landmark_points = []

        for landmark in landmarks.landmark:
            x = min(int(landmark.x * img_width), img_width - 1)
            y = min(int(landmark.y * img_height), img_height - 1)

            landmark_points.append([x, y])

        return landmark_points

    def _cals_bounding_box(self, landmarks_list):
        """Calculates the bounding box for the given landmarks list."""
        landmark_array = np.array(landmarks_list)

        x, y, w, h = cv2.boundingRect(landmark_array)
        x = int(x - 50)
        y = int(y - 50)
        w = int(w + 100)
        h = int(h + 100)

        return [x, y, x + w, y + h]

    def draw_text_info(
        self,
        image,
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
        text_info = "Gesture"
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
    "Application execution."
    landmarks_detector = MediaPipeHandLandmarks(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    landmarks_detector.get_landmarks_from_stream(0)


if __name__ == "__main__":
    main()
