import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set a random seed to ensure different random selections each time
np.random.seed()

# Assuming the order of classes is ['lilly', 'lotus', 'orchid', 'sunflower', 'tulip']
class_labels = ['lilly', 'lotus', 'orchid', 'sunflower', 'tulip']

# Create a class map using the class labels
class_map = {index: label for index, label in enumerate(class_labels)}

# Load your trained model
model_path = 'Flower_Classification/model/best_model.h5'
model = tf.keras.models.load_model(model_path)

# Streamlit app
st.title("Flowers Image Classifier")

# Upload a new image
uploaded_image = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Make predictions on the uploaded image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Make a prediction
    prediction = model.predict(img)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    # Get the predicted class name using the mapping function
    predicted_class_name = class_map.get(predicted_class_index, "Unknown")

    # Display the uploaded and predicted images in two columns
    col1, col2 = st.columns(2)

    # Display the uploaded image in the first column
    col1.image(plt.imread(uploaded_image), caption="Uploaded Image", use_column_width=True)

    # Display the prediction result in the second column with the real class name

    # Display the predicted image in the second column
    col2.image(plt.imread(uploaded_image), caption="Predicted Image", use_column_width=True)
    col2.title(f"Predicted Class Name: {predicted_class_name}")

    # Disable the warning for st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Adjust layout for better spacing
    plt.tight_layout()

# Run the Streamlit app
st.write("Upload an image to classify.")
