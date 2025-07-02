import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your model (cached for performance)
@st.cache(allow_output_mutation=True)
def load_emotion_model():
    model = load_model("model.h5", compile=False)  # Safe if model was saved without compile
    return model

model = load_emotion_model()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit UI
st.title("Facial Emotion Recognition")
st.write("Upload a face image (ideally 48x48 grayscale) or a clear photo with a visible face.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected. Please upload a clearer image.")
        else:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                face = np.reshape(face, (1, 48, 48, 1))

                prediction = model.predict(face)
                emotion = emotion_labels[np.argmax(prediction)]

                # Draw box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Predicted Emotion: {emotion}", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
