from tensorflow import keras
from tensorflow.lite import TFLiteConverter

def save_as_tflite(model_path: str, save_path:str) -> None:
        converter = keras.models.load_model(model_path)
        converter = TFLiteConverter.from_keras_model(converter)
        tflite_model = converter.convert()
        with open(save_path, "wb") as f:
            f.write(tflite_model)