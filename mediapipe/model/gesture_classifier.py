import numpy as np
import tensorflow as tf

class GestureClassifier:
    def __init__(self, model_path: str = "model/gesture_classifier.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def __call__(self, norm_landmarks_list):
        norm_landmarks_list = np.array([norm_landmarks_list], dtype=np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], norm_landmarks_list)
        
        self.interpreter.invoke()
        
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        result_index = np.argmax(result)
        
        return result_index
