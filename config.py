"""
Configuration file for Facial Emotion Recognition project
"""

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (48, 48, 1),
    'num_classes': 7,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'test_split': 0.2
}

# Data Configuration
DATA_CONFIG = {
    'dataset_path': 'data/fer2013.csv',
    'image_size': (48, 48),
    'emotions': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    'emotion_map': {i: emotion for i, emotion in enumerate(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])}
}

# Training Configuration
TRAINING_CONFIG = {
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'min_lr': 1e-7,
    'l2_regularization': 0.01,
    'dropout_rate_conv': 0.25,
    'dropout_rate_dense': 0.5
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Model Architecture Configuration
ARCHITECTURE_CONFIG = {
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}
    ],
    'dense_layers': [
        {'units': 512, 'activation': 'relu'},
        {'units': 256, 'activation': 'relu'}
    ],
    'pool_size': (2, 2),
    'batch_normalization': True
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'training_accuracy': 0.8613,  # 86.13%
    'validation_accuracy': 0.6239,  # 62.39%
    'test_accuracy': 0.60,  # Target test accuracy
    'training_time': 3600,  # Maximum training time in seconds
    'model_size_mb': 50  # Maximum model size in MB
}

# File Paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'analysis_dir': 'data_analysis',
    'processed_images_dir': 'processed_images',
    'reports_dir': 'reports',
    'model_file': 'models/emotion_recognition_model.h5',
    'best_model_file': 'models/best_model.h5',
    'training_history_file': 'data_analysis/training_history.png',
    'confusion_matrix_file': 'data_analysis/confusion_matrix.png',
    'emotion_distribution_file': 'data_analysis/emotion_distribution.png',
    'sample_images_file': 'data_analysis/sample_images.png'
}

# Streamlit App Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Facial Emotion Recognition',
    'page_icon': '😊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'camera_index': 0,
    'confidence_threshold': 0.5,
    'max_image_size': 800
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': (15, 5),
    'dpi': 300,
    'color_palette': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'],
    'emotion_colors': {
        'Angry': '#ff6b6b',
        'Disgust': '#4ecdc4',
        'Fear': '#45b7d1',
        'Happy': '#96ceb4',
        'Sad': '#feca57',
        'Surprise': '#ff9ff3',
        'Neutral': '#54a0ff'
    }
}

# Face Detection Configuration
FACE_DETECTION_CONFIG = {
    'scale_factor': 1.1,
    'min_neighbors': 4,
    'min_size': (30, 30),
    'max_size': None
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/emotion_recognition.log'
}

# Random Seeds
RANDOM_SEEDS = {
    'numpy': 42,
    'tensorflow': 42,
    'python': 42
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'gpu_memory_growth': True,
    'mixed_precision': False,
    'num_workers': 4,
    'use_gpu': True
}

def get_config():
    """
    Get complete configuration dictionary
    """
    return {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'architecture': ARCHITECTURE_CONFIG,
        'performance_targets': PERFORMANCE_TARGETS,
        'paths': PATHS,
        'streamlit': STREAMLIT_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'face_detection': FACE_DETECTION_CONFIG,
        'logging': LOGGING_CONFIG,
        'random_seeds': RANDOM_SEEDS,
        'hardware': HARDWARE_CONFIG
    }

def setup_environment():
    """
    Setup environment variables and configurations
    """
    import os
    import tensorflow as tf
    
    # Set random seeds
    import numpy as np
    np.random.seed(RANDOM_SEEDS['numpy'])
    tf.random.set_seed(RANDOM_SEEDS['tensorflow'])
    
    # GPU configuration
    if HARDWARE_CONFIG['gpu_memory_growth']:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    # Mixed precision
    if HARDWARE_CONFIG['mixed_precision']:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    
    # Create directories
    for path in PATHS.values():
        if isinstance(path, str) and '/' in path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print("Environment setup completed")

if __name__ == "__main__":
    setup_environment()
    print("Configuration loaded successfully") 