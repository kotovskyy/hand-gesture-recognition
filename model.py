import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras
import time

# Define paths to your training and validation data
train_data_dir = 'data/train'
validation_data_dir = 'data/test'

# Define parameters
img_width, img_height = 100, 100  # Input image dimensions
batch_size = 32
epochs = 20
num_classes = 4  # Number of classes in your dataset

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    'inferred',
    label_mode='categorical',
    batch_size = batch_size,
    image_size=(img_width, img_height))

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    'inferred',
    label_mode='categorical',
    batch_size = batch_size,
    image_size=(img_width, img_height))


size = 4
lr = 0.001

dropout_sets = [
    [0.3, 0.3, 0.4, 0.5],
    [0.1, 0.2, 0.2, 0.2],
    [0.4, 0.2, 0.2, 0.2],
    [0.1, 0.2, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2]
]

dropout_set = dropout_sets[0]



model = Sequential()
NAME = f"ClassicModel_EarlyStop3_{lr}lr_{batch_size}bs_{size}layers_{int(time.time())}"
print(f"Model size: {size}")
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout_set[0]))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout_set[1]))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout_set[2]))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout_set[3]))

# Flatten the output
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer (assuming 10 gesture classes)
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


print(model.summary())
tensorboad = TensorBoard(log_dir='logs/{}'.format(NAME))
# Train the model

model_checkpoint = ModelCheckpoint('model2.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    callbacks=[tensorboad, model_checkpoint])

# Evaluate the model
score = model.evaluate(validation_dataset)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
