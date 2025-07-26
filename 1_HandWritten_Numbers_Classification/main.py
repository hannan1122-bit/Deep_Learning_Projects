import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model('Hand_Written_Text_Classifier_model.h5')

# Load and preprocess image
img = cv2.imread("Image_To_Classify.png", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28
img = cv2.resize(img, (28, 28))

# Invert colors (MNIST is white digit on black background)
img = 255 - img

# Normalize to [0, 1]
img = img / 255.0

# Reshape
img = img.reshape(1, 28, 28)

# Prediction
prediction = model.predict(img)
print("Text in the attached Image is:", np.argmax(prediction))
