import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from model import EmotionCNN
from utils import preprocess_face_for_model, create_emotion_visualization
from data_loader import create_sample_data

def create_demo_image():
    """
    Create a demo image with synthetic faces for testing
    """
    # Create a simple demo image (you can replace this with a real image)
    image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Add some "face-like" regions for demo
    # Face 1
    cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(image, (130, 130), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(image, (170, 130), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(image, (150, 160), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Face 2
    cv2.rectangle(image, (350, 150), (450, 250), (255, 255, 255), -1)
    cv2.circle(image, (380, 180), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(image, (420, 180), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(image, (400, 210), (25, 15), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    return image

def test_model_with_sample_data():
    """
    Test the model with sample data
    """
    print("=== Testing Model with Sample Data ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize model
    model = EmotionCNN()
    model.build_model()
    model.compile_model()
    
    # Test predictions on sample data
    X_test = data['X_test'][:10]  # Use first 10 test samples
    
    print(f"Testing on {len(X_test)} sample images...")
    
    predictions = []
    for i, image in enumerate(X_test):
        prediction = model.predict_emotion(image)
        predictions.append(prediction)
        print(f"Sample {i+1}: {prediction['emotion']} (Confidence: {prediction['confidence']:.2%})")
    
    return predictions

def test_face_detection():
    """
    Test face detection functionality
    """
    print("\n=== Testing Face Detection ===")
    
    # Create demo image
    demo_image = create_demo_image()
    
    # Detect faces
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
        cv2.cvtColor(demo_image, cv2.COLOR_BGR2GRAY), 1.1, 4
    )
    
    print(f"Detected {len(faces)} faces in demo image")
    
    # Draw face detections
    result_image = demo_image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save demo image
    cv2.imwrite('demo_face_detection.jpg', result_image)
    print("Demo image with face detections saved as 'demo_face_detection.jpg'")
    
    return faces, demo_image

def test_emotion_recognition_pipeline():
    """
    Test the complete emotion recognition pipeline
    """
    print("\n=== Testing Complete Emotion Recognition Pipeline ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize model
    model = EmotionCNN()
    model.build_model()
    model.compile_model()
    
    # Test on a few samples
    test_images = data['X_test'][:5]
    
    print("Testing emotion recognition on sample images:")
    for i, image in enumerate(test_images):
        # Predict emotion
        prediction = model.predict_emotion(image)
        
        print(f"Image {i+1}:")
        print(f"  Predicted Emotion: {prediction['emotion']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  All Probabilities: {prediction['probabilities']}")
        print()
    
    return model

def create_visualization_demo():
    """
    Create visualization demos
    """
    print("\n=== Creating Visualization Demos ===")
    
    # Create sample predictions
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    sample_predictions = [{
        'emotion': 'Happy',
        'confidence': 0.85,
        'probabilities': [0.05, 0.02, 0.01, 0.85, 0.03, 0.02, 0.02]
    }]
    
    # Create visualization
    fig = create_emotion_visualization(sample_predictions, emotions)
    if fig:
        plt.savefig('demo_emotion_visualization.png', dpi=300, bbox_inches='tight')
        print("Emotion visualization saved as 'demo_emotion_visualization.png'")
        plt.show()

def run_comprehensive_demo():
    """
    Run a comprehensive demo of all features
    """
    print("🎭 Facial Emotion Recognition Demo")
    print("=" * 50)
    
    # Test 1: Model with sample data
    predictions = test_model_with_sample_data()
    
    # Test 2: Face detection
    faces, demo_image = test_face_detection()
    
    # Test 3: Complete pipeline
    model = test_emotion_recognition_pipeline()
    
    # Test 4: Visualizations
    create_visualization_demo()
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Demo Completed Successfully!")
    print("\nDemo Summary:")
    print("- Model tested with sample data")
    print("- Face detection tested")
    print("- Emotion recognition pipeline tested")
    print("- Visualizations created")
    print("\nNext Steps:")
    print("1. Train the model: python train_model.py")
    print("2. Run the web app: streamlit run app.py")
    print("3. Upload the FER-2013 dataset to data/ directory for full training")

def main():
    """
    Main demo function
    """
    try:
        run_comprehensive_demo()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("This is expected if the model hasn't been trained yet.")
        print("Please run 'python train_model.py' first to train the model.")

if __name__ == "__main__":
    main() 