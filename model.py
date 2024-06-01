import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import keras
import time

# Define paths to your training and validation data
train_data_dir = 'data/train'
validation_data_dir = 'data/test'

# Define parameters
img_width, img_height = 100, 100  # Input image dimensions
batch_size = 64
epochs = 10
num_classes = 4  # Number of classes in your dataset

# Preprocess and augment your training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Preprocess your validation data
validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


size = 4
lr = 0.001

dropout_sets = [
    [0.2, 0.2, 0.2, 0.2],
    [0.1, 0.2, 0.2, 0.2],
    [0.4, 0.2, 0.2, 0.2],
    [0.1, 0.2, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2]
]

for i, dropout_set in enumerate(dropout_sets):
    model = Sequential()
    NAME = f"CustomModel_L1reg_{lr}lr_{batch_size}bs_{size}layers_{i}dropset_{int(time.time())}"
    print(f"Model size: {size}")
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l1(), input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_set[0]))
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l1()))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_set[1]))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l1()))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_set[2]))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l1()))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_set[3]))
    
    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Output layer (assuming 10 gesture classes)
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print(model.summary())
    tensorboad = TensorBoard(log_dir='logs/{}'.format(NAME))
    # Train the model
    history = model.fit(
        train_generator,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[tensorboad])

    # Evaluate the model
    score = model.evaluate(validation_generator, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
