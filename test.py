import os
# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from mtcnn import MTCNN
import pickle as pkl
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

print("Loading the saved features....")

filenames = pkl.load(open('filenames.pkl', 'rb'))
embeddings = np.array(pkl.load(open('embeddings.pkl', 'rb')))

# Input image
sample_img_path = 'C:/Users/RISHAV/OneDrive/Desktop/Python/Projects/Project 14-Which Bollywood Celebrity Are You/samples/Salman_khan_duplicate.png'

try:
    print(f"Analyzing input image: {sample_img_path}...")

    detector = MTCNN()
    # load img -> face detection

    sample_img = cv2.imread(sample_img_path)
    results = detector.detect_faces(sample_img)

    x, y, width, height = results[0]['box']

    face = sample_img[y:y+height, x:x+width]

    # cv2.imshow('output.jpg', face)
    # cv2.waitKey(0)

    # embedding the face

    embedder = DeepFace.represent(
        img_path=face,
        model_name='VGG-Face',
        enforce_detection=False,
        max_faces=1
    )
    sample_embedding = embedder[0]['embedding']  # sample image is converted into vectors
    # print(len(sample_embedding))
    # print(embeddings.shape)

    print("Comparing features with 8629 database images...")

    sample_embedding_2d = np.array(sample_embedding).reshape(1, -1)

    # calculating the cosine distance
    similarity_scores = cosine_similarity(sample_embedding_2d, embeddings)[0]

    best_match_index = np.argmax(similarity_scores)
    max_score = similarity_scores[best_match_index]

    best_match_path = filenames[best_match_index]

    actor_folder_name = os.path.basename(os.path.dirname(best_match_path))
    actor_name_clean = actor_folder_name.replace("_", " ")

    print("\n" + "=" * 30)
    print("🎯 PREDICTION SUCCESSFUL!")
    print("=" * 30)
    print(f"🎭 Predicted Celebrity: {actor_name_clean}")
    print(f"📈 Similarity Score: {max_score:.4f} (Higher is better)")
    print(f"📁 Matched Database Image: {best_match_path}")
    print("=" * 30)
    #
    preview = cv2.imread(best_match_path)
    # 2. Check if the image loaded successfully
    if preview is None:
        print("Error: Could not load matched image.")
    else:
        cv2.imshow(f'{best_match_path}', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except ValueError as ve:
    print(f"/n❌ Face Detection Error: Input image is not clear! ({ve})")
except Exception as e:
    print(f"/n❌ Error occurred: {e}")