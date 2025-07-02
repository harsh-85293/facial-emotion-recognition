import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your model
@st.cache_resource
def load_emotion_model():
    return load_model("model.h5")

model = load_emotion_model()

# Emotion labels (update if different in your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# UI
st.title("Facial Emotion Recognition")
st.write("Upload a face image (48x48 grayscale) or a clear photo with a face.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please upload a clear image with a visible face.")
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            resized = cv2.resize(roi_gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            prediction = model.predict(reshaped)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Predicted Emotion: {emotion}", use_column_width=True)
