import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FERDataLoader:
    def __init__(self, data_path='data/fer2013.csv'):
        self.data_path = data_path
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_map = {i: emotion for i, emotion in enumerate(self.emotions)}
        
    def load_data(self):
        """
        Load FER-2013 dataset from CSV file
        """
        try:
            df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Dataset file not found at {self.data_path}")
            print("Please download the FER-2013 dataset and place it in the data/ directory")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset
        """
        # Extract pixels and emotions
        pixels = df['pixels'].values
        emotions = df['emotion'].values
        
        # Convert pixels to images
        images = []
        for pixel_sequence in pixels:
            # Split pixel string and convert to integers
            pixel_array = np.array(pixel_sequence.split(' '), dtype=np.uint8)
            # Reshape to 48x48
            image = pixel_array.reshape(48, 48)
            images.append(image)
        
        images = np.array(images)
        
        # Normalize images
        images = images.astype('float32') / 255.0
        
        # Add channel dimension
        images = np.expand_dims(images, axis=-1)
        
        # Convert emotions to categorical
        emotions_categorical = to_categorical(emotions, num_classes=7)
        
        print(f"Preprocessed data shape: {images.shape}")
        print(f"Emotions shape: {emotions_categorical.shape}")
        
        return images, emotions_categorical
    
    def split_data(self, images, emotions, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, emotions, test_size=test_size, random_state=42, stratify=emotions
        )
        
        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def analyze_data_distribution(self, emotions):
        """
        Analyze the distribution of emotions in the dataset
        """
        emotion_counts = np.sum(emotions, axis=0)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.emotions, y=emotion_counts)
        plt.title('Emotion Distribution in Dataset')
        plt.xlabel('Emotions')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data_analysis/emotion_distribution.png')
        plt.show()
        
        print("Emotion distribution:")
        for i, emotion in enumerate(self.emotions):
            print(f"{emotion}: {emotion_counts[i]}")
    
    def visualize_samples(self, images, emotions, num_samples=16):
        """
        Visualize sample images from the dataset
        """
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            emotion_idx = np.argmax(emotions[i])
            emotion_name = self.emotions[emotion_idx]
            
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            axes[i].set_title(f'{emotion_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_analysis/sample_images.png')
        plt.show()
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Create data generators with augmentation for training
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def load_and_prepare_data(self):
        """
        Complete data loading and preparation pipeline
        """
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Preprocess data
        images, emotions = self.preprocess_data(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(images, emotions)
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators(X_train, y_train, X_val, y_val)
        
        return {
            'train_generator': train_generator,
            'val_generator': val_generator,
            'X_test': X_test,
            'y_test': y_test,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }

def create_sample_data():
    """
    Create a small sample dataset for testing if FER-2013 is not available
    """
    print("Creating sample dataset for testing...")
    
    # Create sample images (random noise for testing)
    num_samples = 1000
    images = np.random.rand(num_samples, 48, 48, 1)
    
    # Create random emotion labels
    emotions = np.random.randint(0, 7, num_samples)
    emotions_categorical = to_categorical(emotions, num_classes=7)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, emotions_categorical, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Sample dataset created:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    } 