#!/usr/bin/env python3
"""
Setup script for Facial Emotion Recognition project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'data',
        'models',
        'data_analysis',
        'processed_images',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_sample_data():
    """Download sample data if FER-2013 is not available"""
    print("\n📊 Setting up sample data...")
    
    # Check if FER-2013 dataset exists
    if not os.path.exists('data/fer2013.csv'):
        print("⚠️  FER-2013 dataset not found")
        print("📝 The system will use sample data for demonstration")
        print("💡 To use the full dataset:")
        print("   1. Download FER-2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("   2. Place fer2013.csv in the data/ directory")
        print("   3. Run: python train_model.py")
    else:
        print("✅ FER-2013 dataset found")

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing imports...")
    
    required_packages = [
        'tensorflow',
        'keras',
        'streamlit',
        'opencv-python',
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas',
        'scikit-learn',
        'pillow'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def test_gpu():
    """Test GPU availability"""
    print("\n🖥️  Testing GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
        
        return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def run_demo():
    """Run a quick demo to test the system"""
    print("\n🎭 Running demo...")
    
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        print("✅ Demo completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if the model hasn't been trained yet")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Train the model:")
    print("   python train_model.py")
    print("\n2. Run the web application:")
    print("   streamlit run app.py")
    print("\n3. For real-time emotion recognition:")
    print("   - Open the web app")
    print("   - Select 'Real-time Webcam' mode")
    print("   - Click 'Start Camera'")
    print("\n4. For image upload:")
    print("   - Select 'Upload Image' mode")
    print("   - Upload an image with faces")
    print("\n5. View model information:")
    print("   - Select 'Model Information' mode")
    
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- config.py: Configuration settings")
    print("- model.py: CNN architecture")
    print("- data_loader.py: Data processing")
    
    print("\n🔧 Troubleshooting:")
    print("- If training fails, check GPU availability")
    print("- If webcam doesn't work, try different camera index")
    print("- For dataset issues, see data_loader.py")
    
    print("\n🎯 Performance Targets:")
    print("- Training Accuracy: 86.13%")
    print("- Validation Accuracy: 62.39%")
    print("- Epochs: 50")
    print("- Dataset: FER-2013 (30,000+ images)")

def main():
    """Main setup function"""
    print("🎭 Facial Emotion Recognition Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Download sample data
    download_sample_data()
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test GPU
    test_gpu()
    
    # Run demo
    run_demo()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 