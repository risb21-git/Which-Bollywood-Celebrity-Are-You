# 🎭 Which Bollywood Celebrity Are You?

A Deep Learning based celebrity look-alike finder that predicts which Bollywood celebrity resembles the uploaded image the most.

## 🚀 Features

* Upload an image through a Streamlit web interface
* Detect faces using MTCNN
* Extract facial embeddings using DeepFace (VGG-Face)
* Compare the uploaded face with a celebrity database
* Find the most similar Bollywood celebrity using Cosine Similarity
* Display the matched celebrity image

## 🛠️ Technologies Used

* Python
* Streamlit
* OpenCV
* MTCNN
* DeepFace
* NumPy
* Scikit-Learn
* Pickle

## 📂 Project Structure

```text
├── app.py
├── embeddings.pkl
├── filenames.pkl
├── uploads/
├── celebrity_dataset/
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd Which-Bollywood-Celebrity-Are-You
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 📌 Note

The files `embeddings.pkl` and `filenames.pkl` are not included in this repository because of GitHub file size limitations.

To run the project, generate these files using the feature extraction script and celebrity image dataset.

## 🧠 How It Works

1. User uploads an image.
2. MTCNN detects and crops the face.
3. DeepFace (VGG-Face) generates a facial embedding.
4. Cosine Similarity compares the embedding against the celebrity database.
5. The celebrity with the highest similarity score is returned.

## 📷 Demo

Upload a face image and the system predicts the most similar Bollywood celebrity.

## 👨‍💻 Author

Rishav
B.Sc. in Data Science
Aspiring Data Scientist
