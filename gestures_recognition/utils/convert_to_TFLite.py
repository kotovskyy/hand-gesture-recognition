from tensorflow import keras
import tensorflow as tf

def save_as_tflite(model_path: str, save_path:str) -> None:
        model = keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(save_path, "wb") as f:
            f.write(tflite_model)
            
save_as_tflite('models/sign_net-1717946829.hdf5', 'models/sign_net_v6.tflite')
