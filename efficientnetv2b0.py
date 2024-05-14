from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def test_model():
    # Define paths to your training and validation data
    train_data_dir = 'data/train'
    validation_data_dir = 'data/test'

    # Define parameters
    img_width, img_height = 100, 100  # Input image dimensions
    batch_size = 32
    epochs = 3
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


    base_model = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_width, img_height, 3)
    )

    # Build your model on top of the pre-trained VGG19 model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

    # Evaluate the model
    score = model.evaluate(validation_generator, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def load_and_preprocess_image(img_path, img_width, img_height):
    img = keras.utils.load_img(img_path, target_size=(img_width, img_height))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_and_preprocess_images(image_dir, img_width=100, img_height=100):
    import os
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
            img_path = os.path.join(image_dir, filename)
            x = load_and_preprocess_image(img_path, img_width, img_height)
            images.append(x)
    return images

def test_image():
    LABELS = ["A", "F", "L", "Y"]
    img_path = "data/train/A/1_image0.png"
    img_width, img_height = 100, 100
    num_classes = 4

    # img = keras.utils.load_img(img_path, target_size=(img_width, img_height))
    # x = keras.utils.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # print(x.shape)
    
    images = load_and_preprocess_images("data/train/A")
    
    # base_model = EfficientNetV2B0(
    #     weights='imagenet',
    #     include_top=False,
    #     input_shape=(img_width, img_height, 3),
    #     classes=4
    # )
    
    # model = Sequential()
    # model.add(base_model)
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(4, activation='softmax'))
    
    
    # model.summary()
    
    # model.compile(
    #     optimizer=keras.optimizers.SGD(0.0001, 0.9),
    #     loss=keras.metrics.categorical_crossentropy,
    #     metrics=['accuracy']
    # )
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)
    
    
    model = keras.models.Sequential()
    model.add(layers.Input(shape=(100, 100, 3)))
    # model.add(data_augmentation)
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.64))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=4, activation='softmax'))
    
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.metrics.categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    
    train_data_dir = 'data/train'
    validation_data_dir = 'data/test'
    batch_size = 64
    epochs = 3
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


    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

    model.save_weights("model_weights_shakal_10epochs.h5")

    counter = 0
    for image in images[:100]:
        features = model.predict(image)
        result = LABELS[np.argmax(features)]
        if result == "A": 
            counter += 1
        print(f"Features : {features}")
        print(f"Result : {result}")
    
    print(f"Accuracy: {counter/100}")



def main():
    test_image()

if __name__ == "__main__":
    main()