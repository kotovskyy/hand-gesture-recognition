from tensorflow import keras
from tensorflow.lite import TFLiteConverter
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from typing import Tuple
from keras.utils import image_dataset_from_directory
from keras.callbacks import ModelCheckpoint, TensorBoard
import time

SEED = 111

class SignNet:
    def __init__(
        self, input_shape: Tuple[int] = (224, 224, 3), n_classes: int = 5
    ) -> None:
        self.image_shape = input_shape

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(n_classes, activation="softmax"))

        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def train(
        self,
        dataset_path,
        batch_size: int = 32,
        epochs: int = 10,
        validation_split: int = 0.2,
        additional_data: str = "",
    ) -> None:

        train_dataset, validation_dataset = image_dataset_from_directory(
            dataset_path,
            subset="both",
            label_mode="int",
            image_size=self.image_shape[:2],
            batch_size=batch_size,
            validation_split=validation_split,
            seed=SEED,
        )

        name = f"sign_net-{int(time.time())}{additional_data}"
        tensorboard = TensorBoard(log_dir=f"logs/{name}")
        model_checkpoint = ModelCheckpoint(
            filepath=f"models/{name}.hdf5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        )

        self.model.fit(
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_dataset,
            callbacks=[tensorboard, model_checkpoint],
        )


if __name__ == "__main__":
    sign_net = SignNet(n_classes=3)
    sign_net.train("data", epochs=10, validation_split=0.2)
