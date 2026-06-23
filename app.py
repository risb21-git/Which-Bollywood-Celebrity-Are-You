import os
# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
import cv2
from mtcnn import MTCNN
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

detector = MTCNN()

embeddings = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True

    except:
        return False

def extract_features(img_path, detector):
    img = cv2.imread(img_path)

    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x, y, width, height = results[0]['box']

    x = max(0, x)
    y = max(0, y)

    face = img[y:y+height, x:x+width]

    try:
        embedding = DeepFace.represent(
            img_path=face,
            model_name='VGG-Face',
            enforce_detection=False
        )
    except Exception:
        return None

    feature_vector = embedding[0]['embedding']

    return np.array(feature_vector).reshape(1, -1)

def recommend(features, embeddings):
    similarity_scores = cosine_similarity(
        features,
        embeddings
    )[0]

    best_match_index = np.argmax(similarity_scores)

    return best_match_index

st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(
            os.path.join('uploads', uploaded_image.name),
            detector
        )

        if features is None:
            st.error("No face detected in the uploaded image.")
            st.stop()

        index_pos = recommend(features, embeddings)

        matched_path = filenames[index_pos]

        actor_folder = os.path.basename(
            os.path.dirname(matched_path)
        )

        predicted_actor = actor_folder.replace("_", " ")
        # display
        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image, width=300)
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos],width=300)