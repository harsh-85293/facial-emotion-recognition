# Facial Emotion Recognition Using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) for facial emotion recognition using the FER-2013 dataset. The model achieves 86.13% training accuracy and 62.39% validation accuracy over 50 epochs with innovative regularization techniques.

## Features
- **CNN Model**: Built with TensorFlow/Keras for emotion classification
- **Dataset**: FER-2013 dataset with 30,000+ images across 7 emotions
- **Regularization**: Advanced techniques for model stability
- **Real-time Recognition**: Streamlit web application with webcam interface
- **Performance**: High accuracy with robust validation

## Emotions Classified
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd facial-emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python train_model.py
```

### Running the Streamlit App
```bash
streamlit run app.py
```

## Project Structure
```
facial-emotion-recognition/
├── data/                   # Dataset directory
├── models/                 # Trained models
├── utils/                  # Utility functions
├── train_model.py         # Training script
├── app.py                 # Streamlit application
├── model.py               # CNN model architecture
├── data_loader.py         # Data preprocessing
└── requirements.txt       # Dependencies
```

## Model Architecture
- Convolutional layers with batch normalization
- Dropout for regularization
- Dense layers with activation functions
- Optimized for emotion classification

## Performance Metrics
- Training Accuracy: 86.13%
- Validation Accuracy: 62.39%
- Epochs: 50
- Dataset: FER-2013 (30,000+ images)

## Technologies Used
- TensorFlow/Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Pandas

## License
MIT License 