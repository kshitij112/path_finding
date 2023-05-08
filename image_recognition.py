import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the pre-trained model
model = keras.applications.ResNet50(weights='imagenet')

# Load and preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Make predictions on the image
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Example usage
image_path = r'C:\Users\kshitij\Dropbox\My PC (LAPTOP-BUA9FSN7)\Documents\Scanned Documents\Welcome Scan.jpg'
predictions = predict_image(image_path)

# Print the top predictions
for pred in predictions:
    print(f"{pred[1]}: {pred[2]*100}%")

