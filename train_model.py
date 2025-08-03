import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from model import EmotionCNN, create_callbacks
from data_loader import FERDataLoader, create_sample_data

def create_directories():
    """
    Create necessary directories for the project
    """
    directories = ['models', 'data_analysis', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('data_analysis/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('data_analysis/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the trained model
    """
    # Predict on test set
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Calculate accuracy
    test_accuracy = np.mean(y_pred == y_true)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return test_accuracy

def main():
    """
    Main training function
    """
    print("=== Facial Emotion Recognition CNN Training ===")
    
    # Create directories
    create_directories()
    
    # Initialize data loader
    data_loader = FERDataLoader()
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    data = data_loader.load_and_prepare_data()
    
    # If FER-2013 dataset is not available, create sample data
    if data is None:
        print("FER-2013 dataset not found. Using sample data for demonstration...")
        data = create_sample_data()
    
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Analyze data distribution
    print("\nAnalyzing data distribution...")
    data_loader.analyze_data_distribution(y_train)
    
    # Visualize sample images
    print("\nVisualizing sample images...")
    data_loader.visualize_samples(X_train, y_train)
    
    # Initialize and build model
    print("\nBuilding CNN model...")
    model = EmotionCNN()
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Print model summary
    print("\nModel Architecture:")
    model.get_model_summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Training parameters
    epochs = 50
    batch_size = 32
    
    print(f"\nStarting training for {epochs} epochs...")
    print("Expected performance targets:")
    print("- Training Accuracy: 86.13%")
    print("- Validation Accuracy: 62.39%")
    
    # Train the model
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    test_accuracy = evaluate_model(model.model, X_test, y_test, class_names)
    
    # Save the trained model
    print("\nSaving trained model...")
    model.save_model('models/emotion_recognition_model.h5')
    
    # Print final results
    print("\n=== Training Complete ===")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save training metrics
    metrics = {
        'final_training_accuracy': history.history['accuracy'][-1],
        'final_validation_accuracy': history.history['val_accuracy'][-1],
        'test_accuracy': test_accuracy,
        'epochs_trained': len(history.history['accuracy']),
        'target_training_accuracy': 0.8613,
        'target_validation_accuracy': 0.6239
    }
    
    print("\nPerformance Summary:")
    print(f"Target Training Accuracy: 86.13%")
    print(f"Achieved Training Accuracy: {metrics['final_training_accuracy']:.2%}")
    print(f"Target Validation Accuracy: 62.39%")
    print(f"Achieved Validation Accuracy: {metrics['final_validation_accuracy']:.2%}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
    
    return model, metrics

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    model, metrics = main() 