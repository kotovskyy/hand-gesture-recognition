from model import Gesture_Classifier

# Create an instance of the Gesture_Classifier class
classifier = Gesture_Classifier(image_height=128, image_width=128, num_channels=3, num_classes=6)

# Display the model summary
classifier.summary()
