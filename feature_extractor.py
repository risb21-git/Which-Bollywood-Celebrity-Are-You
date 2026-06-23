# import os
import pickle
#
# actors = os.listdir('data')
#
# filenames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data', actor)):
#         filenames.append(os.path.join('data', actor, file))
# print(len(filenames))
# pickle.dump(filenames, open('filenames.pkl', 'wb'))

from deepface import DeepFace
from tqdm import tqdm

# Loading the model
model_name = "VGG-Face"
model = DeepFace.build_model(model_name)

print(f"{model_name} model loaded successfully!")
# print(model.model.summary())

# Extracting feaure
# 1. Saved file paths ko wapas load karna
print("Loading filenames...")
filenames = pickle.load(open('filenames.pkl', 'rb'))

# 2. Ek khali list jahan humari saari actors ki photos ke features (numbers) save honge
feature_list = []

print(f"Total {len(filenames)} photos found. Feature Extraction is working...")

# 3. Loop chalayenge har photo ke liye
for file in tqdm(filenames):
    try:
        # enforce_detection=False rakha hai taaki kharab photo par loop crash na ho
        result = DeepFace.represent(
            img_path=file,
            model_name="VGG-Face",
            enforce_detection=False
        )

        # Result me se 'embedding' array nikal kar apni list mein daalenge
        embedding = result[0]['embedding']
        feature_list.append(embedding)

    except Exception as e:
        # Agar photo mein koi problem aayi toh code usko skip kar dega
        print(f"\nSkipping {file} due to error: {e}")
        pass

# 4. Loop khatam hone ke baad, final features ko 'embeddings.pkl' mein save karenge
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
print("Feature extraction complete! Data is saved into 'embeddings.pkl'.")