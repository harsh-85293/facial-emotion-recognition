import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("Emotion Recognition App")

# Load model and face detector
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.write("Use your webcam to capture an image or upload one.")

# Capture or upload image
img_buffer = st.camera_input("Take a photo") or st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if img_buffer:
    # Read image data
    bytes_data = img_buffer.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.write("No face detected. Please try again.")
    else:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)
            label = labels[np.argmax(preds)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        st.image(frame, channels="BGR", caption="Processed Image")
