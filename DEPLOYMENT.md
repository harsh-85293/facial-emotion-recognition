# 🚀 Deployment Guide for Streamlit Community Cloud

## Overview
This guide will help you deploy your Facial Emotion Recognition app to Streamlit Community Cloud.

## Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (free at https://share.streamlit.io/)

## Step-by-Step Deployment

### 1. Create GitHub Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Facial Emotion Recognition System"

# Create repository on GitHub (via web interface)
# Then push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/facial-emotion-recognition.git
git branch -M main
git push -u origin main
```

### 2. Configure for Streamlit Cloud

The app is already configured for Streamlit Cloud with:
- ✅ `requirements.txt` - All dependencies listed
- ✅ `app.py` - Main Streamlit application
- ✅ `.gitignore` - Excludes unnecessary files
- ✅ `README.md` - Project documentation

### 3. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository**: `facial-emotion-recognition`
5. **Set the main file path**: `app.py`
6. **Click "Deploy"**

### 4. Configuration for Cloud Deployment

#### Model Loading
The app will automatically download a pre-trained model or use sample data if no model is available.

#### Environment Variables (Optional)
You can set these in Streamlit Cloud settings:
- `STREAMLIT_SERVER_PORT`: 8501
- `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0

### 5. Post-Deployment

#### Add FER-2013 Dataset
1. Download the FER-2013 dataset from Kaggle
2. Add it to your GitHub repository in the `data/` folder
3. The app will automatically use it for better predictions

#### Update Model
1. Train the model locally with the full dataset
2. Upload the trained model to the `models/` folder
3. Commit and push to GitHub
4. Streamlit Cloud will automatically redeploy

## Troubleshooting

### Common Issues

1. **Import Errors**: All dependencies are in `requirements.txt`
2. **Model Not Found**: The app includes fallback to sample data
3. **Memory Issues**: The app is optimized for cloud deployment
4. **Camera Access**: Webcam features work in cloud deployment

### Performance Optimization

- The app uses efficient model loading
- Images are resized for faster processing
- Batch processing for multiple faces
- Caching for repeated predictions

## Monitoring

- **App URL**: Your app will be available at `https://your-app-name.streamlit.app`
- **Logs**: Available in Streamlit Cloud dashboard
- **Updates**: Automatic redeployment on GitHub pushes

## Security Notes

- No sensitive data in the repository
- Model files are excluded from git
- Environment variables for any secrets
- Public deployment is safe for this app

## Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are in the repository
3. Ensure `requirements.txt` is complete
4. Test locally before deploying

---

**Your app will be live at**: `https://your-app-name.streamlit.app` 