import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from model import EmotionCNN

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .emotion-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 0.25rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 20px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path='models/best_model.h5'):
    """
    Load the trained emotion recognition model
    """
    
    try:
        if os.path.exists(model_path):
            model = EmotionCNN()
            model.load_model(model_path)
            st.success("✅ Model loaded successfully!")
            return model
        else:
            st.warning("⚠️ Model file not found. Using demo mode with sample predictions.")
            # Create a dummy model for demo purposes
            model = EmotionCNN()
            model.build_model()
            model.compile_model()
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Using demo mode with sample predictions.")
        # Create a dummy model for demo purposes
        model = EmotionCNN()
        model.build_model()
        model.compile_model()
        return model

def preprocess_face_image(image):
    """
    Preprocess face image for emotion recognition
    """
    # Convert to grayscale (handle both BGR and RGB)
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
    else:
        gray = image
    
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    
    # Normalize to [0, 1] range
    normalized = resized.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    processed = np.expand_dims(normalized, axis=[0, -1])
    
    return processed

def detect_faces(image):
    """
    Detect faces in the image using OpenCV
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces

def draw_emotion_on_image(image, faces, predictions):
    """
    Draw emotion predictions on the image
    """
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    colors = {
        'Angry': (0, 0, 255),      # Red
        'Disgust': (0, 255, 0),     # Green
        'Fear': (255, 0, 255),      # Magenta
        'Happy': (0, 255, 255),     # Yellow
        'Sad': (255, 0, 0),         # Blue
        'Surprise': (128, 0, 128),  # Purple
        'Neutral': (128, 128, 128)  # Gray
    }
    
    for i, (x, y, w, h) in enumerate(faces):
        if i < len(predictions):
            emotion = predictions[i]['emotion']
            confidence = predictions[i]['confidence']
            color = colors.get(emotion, (255, 255, 255))
            
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return image

def create_emotion_chart(predictions):
    """
    Create a bar chart showing emotion probabilities
    """
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    if predictions:
        probabilities = predictions[0]['probabilities']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(emotions, probabilities, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'])
        ax.set_title('Emotion Probabilities', fontsize=16, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    return None

def main():
    """
    Main Streamlit application
    """
    # Header
    st.markdown('<h1 class="main-header">😊 Facial Emotion Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Load model
    model = load_model(f'models/{model_choice}')
    if model is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["📹 Real-time Webcam", "📁 Upload Image", "📊 Model Information"]
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", help="Show raw probabilities and model input")
    
    # Model selection for testing
    model_choice = st.sidebar.selectbox(
        "🤖 Model Selection",
        ["best_model.h5", "emotion_recognition_model.h5"],
        help="Choose which model to use for testing"
    )
    
    if mode == "📹 Real-time Webcam":
        real_time_mode(model, debug_mode)
    elif mode == "📁 Upload Image":
        upload_mode(model, debug_mode)
    elif mode == "📊 Model Information":
        model_info_mode()

def real_time_mode(model, debug_mode=False):
    """
    Real-time webcam emotion recognition using Streamlit's browser camera
    """
    st.header("📹 Real-time Emotion Recognition")
    
    # Camera settings
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    
    with col2:
        st.info("📷 Click the camera button below to start")
    
    # Use Streamlit's browser camera input
    img_bytes = st.camera_input("📷 Look into your webcam for emotion recognition")
    
    if img_bytes is not None:
        # Convert camera input to OpenCV format
        file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Detect faces
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            st.warning("👤 No faces detected. Please position your face in the camera view.")
        else:
            st.success(f"✅ Detected {len(faces)} face(s)!")
            
            # Process each face
            predictions = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = preprocess_face_image(face_img)
                
                # Predict emotion
                prediction = model.predict_emotion(processed_face)
                
                # Debug: Print raw probabilities and class mapping
                if debug_mode:
                    st.write("🔍 **Debug - Raw Probabilities:**")
                    
                    # 1) Print the raw predictions
                    raw_preds = prediction['probabilities']
                    st.write("🔍 Raw preds:", raw_preds.tolist())
                    
                    # 2) Print them alongside class names in order
                    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    st.write("📊 **Class Mapping Check:**")
                    for idx, name in enumerate(class_names):
                        st.write(f"{name:>8} → {raw_preds[idx]*100:5.2f}%")
                    
                    # 3) Show the winner
                    best_idx = np.argmax(raw_preds)
                    st.write(f"🎯 **Prediction:** {class_names[best_idx]} ({raw_preds[best_idx]*100:.2f}%)")
                    
                    # Show what the model actually sees
                    st.write("🔍 **Model Input (48x48 grayscale):**")
                    processed_display = processed_face[0, :, :, 0]  # Remove batch and channel dims
                    st.image(processed_display, caption="What the model sees", use_column_width=True)
                    
                    # Show preprocessing steps
                    st.write("🔧 **Preprocessing Steps:**")
                    st.write(f"- Input shape: {face_img.shape}")
                    st.write(f"- Grayscale shape: {cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).shape}")
                    st.write(f"- Resized shape: {cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), (48, 48)).shape}")
                    st.write(f"- Final processed shape: {processed_face.shape}")
                    st.write("---")
                
                # Filter by confidence threshold
                if prediction['confidence'] >= confidence_threshold:
                    predictions.append(prediction)
            
            # Display results
            if predictions:
                # Draw predictions on frame
                result_frame = draw_emotion_on_image(frame.copy(), faces, predictions)
                
                # Show results in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📸 Captured Image with Detections")
                    st.image(result_frame, channels="BGR", use_column_width=True)
                
                with col2:
                    st.subheader("🎭 Emotion Analysis")
                    
                    for i, pred in enumerate(predictions):
                        st.markdown(f"**Face {i+1}:**")
                        
                        # Emotion display
                        emotion_color = {
                            'Angry': '#ff6b6b', 'Disgust': '#4ecdc4', 'Fear': '#45b7d1',
                            'Happy': '#96ceb4', 'Sad': '#feca57', 'Surprise': '#ff9ff3', 'Neutral': '#54a0ff'
                        }
                        
                        st.markdown(f"""
                        <div class="emotion-display" style="background-color: {emotion_color.get(pred['emotion'], '#e0e0e0')}">
                            {pred['emotion']} ({pred['confidence']:.2%})
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        confidence = pred['confidence']
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                    
                    # Emotion chart
                    if predictions:
                        fig = create_emotion_chart(predictions)
                        if fig:
                            st.pyplot(fig)
            else:
                st.warning("⚠️ No emotions detected above the confidence threshold.")
                st.info("💡 Try adjusting the confidence threshold or repositioning your face.")

def upload_mode(model, debug_mode=False):
    """
    Upload image for emotion recognition
    """
    st.header("📁 Upload Image for Emotion Recognition")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing faces"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert PIL to OpenCV format
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array
        
        # Detect faces
        faces = detect_faces(image_cv)
        
        if len(faces) == 0:
            st.warning("No faces detected in the image.")
        else:
            st.success(f"Detected {len(faces)} face(s) in the image.")
            
            # Process each face
            predictions = []
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_img = image_cv[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = preprocess_face_image(face_img)
                
                # Predict emotion
                prediction = model.predict_emotion(processed_face)
                
                # Debug: Print raw probabilities and class mapping
                if debug_mode:
                    st.write("🔍 **Debug - Raw Probabilities:**")
                    
                    # 1) Print the raw predictions
                    raw_preds = prediction['probabilities']
                    st.write("🔍 Raw preds:", raw_preds.tolist())
                    
                    # 2) Print them alongside class names in order
                    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    st.write("📊 **Class Mapping Check:**")
                    for idx, name in enumerate(class_names):
                        st.write(f"{name:>8} → {raw_preds[idx]*100:5.2f}%")
                    
                    # 3) Show the winner
                    best_idx = np.argmax(raw_preds)
                    st.write(f"🎯 **Prediction:** {class_names[best_idx]} ({raw_preds[best_idx]*100:.2f}%)")
                    
                    # Show what the model actually sees
                    st.write("🔍 **Model Input (48x48 grayscale):**")
                    processed_display = processed_face[0, :, :, 0]  # Remove batch and channel dims
                    st.image(processed_display, caption="What the model sees", use_column_width=True)
                    
                    # Show preprocessing steps
                    st.write("🔧 **Preprocessing Steps:**")
                    st.write(f"- Input shape: {face_img.shape}")
                    st.write(f"- Grayscale shape: {cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).shape}")
                    st.write(f"- Resized shape: {cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), (48, 48)).shape}")
                    st.write(f"- Final processed shape: {processed_face.shape}")
                    st.write("---")
                
                predictions.append(prediction)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image with Detections")
                # Draw predictions on image
                result_image = draw_emotion_on_image(image_cv.copy(), faces, predictions)
                st.image(result_image, channels="BGR", use_column_width=True)
            
            with col2:
                st.subheader("📊 Emotion Analysis")
                
                for i, pred in enumerate(predictions):
                    st.markdown(f"**Face {i+1}:**")
                    
                    # Emotion display
                    emotion_color = {
                        'Angry': '#ff6b6b', 'Disgust': '#4ecdc4', 'Fear': '#45b7d1',
                        'Happy': '#96ceb4', 'Sad': '#feca57', 'Surprise': '#ff9ff3', 'Neutral': '#54a0ff'
                    }
                    
                    st.markdown(f"""
                    <div class="emotion-display" style="background-color: {emotion_color.get(pred['emotion'], '#e0e0e0')}">
                        {pred['emotion']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    confidence = pred['confidence']
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # Emotion chart
                if predictions:
                    fig = create_emotion_chart(predictions)
                    if fig:
                        st.pyplot(fig)

def model_info_mode():
    """
    Display model information and performance metrics
    """
    st.header("📊 Model Information")
    
    # Model architecture info
    st.subheader("🏗️ Model Architecture")
    st.markdown("""
    - **Type:** Convolutional Neural Network (CNN)
    - **Framework:** TensorFlow/Keras
    - **Input Shape:** 48x48x1 (grayscale images)
    - **Output:** 7 emotion classes
    """)
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Accuracy", "86.13%")
    
    with col2:
        st.metric("Validation Accuracy", "62.39%")
    
    with col3:
        st.metric("Epochs Trained", "50")
    
    # Dataset information
    st.subheader("📚 Dataset Information")
    st.markdown("""
    - **Dataset:** FER-2013 (Facial Expression Recognition)
    - **Images:** 30,000+ facial expressions
    - **Emotions:** 7 classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    - **Image Size:** 48x48 pixels (grayscale)
    """)
    
    # Regularization techniques
    st.subheader("🔧 Innovative Regularization Techniques")
    st.markdown("""
    - **Batch Normalization:** Applied after each convolutional layer
    - **Dropout:** 25% after pooling layers, 50% in dense layers
    - **L2 Regularization:** Applied to dense layers (λ=0.01)
    - **Data Augmentation:** Rotation, zoom, flip, contrast adjustment
    - **Early Stopping:** Prevents overfitting
    - **Learning Rate Scheduling:** Reduces learning rate on plateau
    """)
    
    # Emotion classes
    st.subheader("😊 Emotion Classes")
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    for emotion in emotions:
        st.markdown(f"- **{emotion}**")

if __name__ == "__main__":
    main() 