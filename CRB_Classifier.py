import streamlit as st
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved autoencoder model
autoencoder = load_model('trained_autoencoder_model.h5')

# Load the saved K-Means model
kmeans = joblib.load('2_kmeans_model.pkl')

# Function to preprocess the new input image
def preprocess_image(image, target_size=(150, 150)):
    img = image.resize(target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the cluster
def predict_cluster(image):
    # Preprocess the new image
    new_image = preprocess_image(image)

    # Encode the new image using the autoencoder
    encoded_image = autoencoder.predict(new_image)

    # Flatten the encoded image for clustering
    encoded_image_flat = encoded_image.reshape((encoded_image.shape[0], np.prod(encoded_image.shape[1:])))

    # Predict the cluster using the trained K-Means model
    cluster_label = kmeans.predict(encoded_image_flat)

    # Map cluster label to result
    if cluster_label[0] == 0:
        result = "Medium"
    elif cluster_label[0] == 1:
        result = "High"
    elif cluster_label[0] == 2:
        result = "Low"

    return result

st.title('CRB Infection Severity Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize the image for display
    display_size = (300, 300)  # Adjust the size as needed
    resized_image = image.resize(display_size)

    st.image(resized_image, caption='Uploaded Image.', use_container_width=False)
    st.write("")
    st.write("Classifying...")

    # Predict the cluster
    result = predict_cluster(image)
    st.write(f"CRB infection is: {result}")
