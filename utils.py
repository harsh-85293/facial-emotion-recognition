import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

def detect_faces_opencv(image):
    """
    Detect faces using OpenCV Haar Cascade
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def preprocess_face_for_model(face_image, target_size=(48, 48)):
    """
    Preprocess face image for emotion recognition model
    """
    # Convert to grayscale if needed
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    processed = np.expand_dims(normalized, axis=[0, -1])
    
    return processed

def draw_face_detections(image, faces, predictions=None):
    """
    Draw face detection boxes and predictions on image
    """
    result_image = image.copy()
    
    for i, (x, y, w, h) in enumerate(faces):
        # Draw rectangle around face
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add prediction label if available
        if predictions and i < len(predictions):
            emotion = predictions[i]['emotion']
            confidence = predictions[i]['confidence']
            label = f"{emotion}: {confidence:.2f}"
            
            # Choose color based on emotion
            colors = {
                'Angry': (0, 0, 255),      # Red
                'Disgust': (0, 255, 0),     # Green
                'Fear': (255, 0, 255),      # Magenta
                'Happy': (0, 255, 255),     # Yellow
                'Sad': (255, 0, 0),         # Blue
                'Surprise': (128, 0, 128),  # Purple
                'Neutral': (128, 128, 128)  # Gray
            }
            
            color = colors.get(emotion, (255, 255, 255))
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return result_image

def create_emotion_visualization(predictions, emotions):
    """
    Create visualization of emotion predictions
    """
    if not predictions:
        return None
    
    probabilities = predictions[0]['probabilities']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    bars = ax1.bar(emotions, probabilities, 
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'])
    ax1.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Probability')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(probabilities, labels=emotions, autopct='%1.1f%%', 
            colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'])
    ax2.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def save_processed_image(image, filename, output_dir='processed_images'):
    """
    Save processed image to directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Image saved to {filepath}")

def load_image_from_path(image_path):
    """
    Load image from file path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    return image

def resize_image_maintaining_aspect_ratio(image, max_size=800):
    """
    Resize image while maintaining aspect ratio
    """
    height, width = image.shape[:2]
    
    if height <= max_size and width <= max_size:
        return image
    
    # Calculate new dimensions
    if height > width:
        new_height = max_size
        new_width = int(width * max_size / height)
    else:
        new_width = max_size
        new_height = int(height * max_size / width)
    
    resized = cv2.resize(image, (new_width, new_height))
    return resized

def create_emotion_summary(predictions):
    """
    Create a summary of emotion predictions
    """
    if not predictions:
        return "No predictions available"
    
    summary = []
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    for i, pred in enumerate(predictions):
        emotion = pred['emotion']
        confidence = pred['confidence']
        summary.append(f"Face {i+1}: {emotion} (Confidence: {confidence:.2%})")
    
    return "\n".join(summary)

def validate_image_format(image):
    """
    Validate image format and dimensions
    """
    if image is None:
        return False, "Image is None"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image dimensions: {image.shape}"
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, f"Invalid number of channels: {image.shape[2]}"
    
    return True, "Image format is valid"

def enhance_image_contrast(image):
    """
    Enhance image contrast using CLAHE
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def create_face_detection_report(image_path, faces, predictions=None):
    """
    Create a comprehensive report of face detection results
    """
    report = {
        'image_path': image_path,
        'image_shape': None,
        'faces_detected': len(faces),
        'face_coordinates': faces.tolist() if len(faces) > 0 else [],
        'predictions': predictions,
        'processing_time': None
    }
    
    # Load image to get shape
    try:
        image = load_image_from_path(image_path)
        report['image_shape'] = image.shape
    except Exception as e:
        report['error'] = str(e)
    
    return report

def save_report_to_file(report, filename='face_detection_report.txt'):
    """
    Save detection report to file
    """
    with open(filename, 'w') as f:
        f.write("Face Detection Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Image Path: {report['image_path']}\n")
        f.write(f"Image Shape: {report['image_shape']}\n")
        f.write(f"Faces Detected: {report['faces_detected']}\n\n")
        
        if report['face_coordinates']:
            f.write("Face Coordinates:\n")
            for i, (x, y, w, h) in enumerate(report['face_coordinates']):
                f.write(f"  Face {i+1}: ({x}, {y}, {w}, {h})\n")
        
        if report['predictions']:
            f.write("\nEmotion Predictions:\n")
            for i, pred in enumerate(report['predictions']):
                f.write(f"  Face {i+1}: {pred['emotion']} (Confidence: {pred['confidence']:.2%})\n")
    
    print(f"Report saved to {filename}") 