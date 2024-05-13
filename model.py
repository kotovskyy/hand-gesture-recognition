from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class Gesture_Classifier:
    def __init__ (self, image_height, image_width, num_channels, num_classes):
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = self._create_model()

    def _create_model(self):
        # Initialize the model
        model = Sequential()

        # Add convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_height, self.image_width, self.num_channels)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Flatten layer
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def summary(self):
        self.model.summary()
