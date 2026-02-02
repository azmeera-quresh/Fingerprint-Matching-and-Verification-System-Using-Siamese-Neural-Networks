# Fingerprint Matching and Verification System

## Authors
- Azmeera Qureshi (2103626)
- Hamza Munzir (2243239)

## Project Overview
This project implements a fingerprint matching system using a Siamese neural network for accurate and efficient fingerprint identification. The system comprises three main stages: image preprocessing, feature extraction, and matching.

## Methodology

### Fingerprint Image Processing
1. **Gray Scaling**: Conversion from color to grayscale to simplify representation and focus on intensity values.
2. **Histogram Equalization**: Enhancement of contrast and normalization of grayscale values to address inconsistent clarity.
3. **Low-pass Filtering**: Noise removal and smoothing using Fast Fourier Transform (FFT).

### Fingerprint Enhancement
- Noise reduction techniques (Gaussian or median filtering)
- Ridge thinning
- Ridge orientation estimation
- Ridge frequency estimation

### Siamese Network Architecture
- Utilizes VGG16 pre-trained weights
- Trained on 63 different fingerprints
- Image size: 98 × 98 (though code uses 90x90)
- Data augmentation: Rotation (5 times per image)
- Training set: 66 different fingerprints, ~10 images each
- Total training samples: 49,270 after augmentation
- Validation split: 10%

## Model Details
- **Architecture**: Siamese network with shared CNN branches
- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Input Shape**: (90, 90, 1) - grayscale images
- **Output**: 4-dimensional feature vector for similarity comparison
- **Loss Function**: Contrastive loss for Siamese networks
- **Optimizer**: Adam
- **Training**: 2 epochs, batch size 32

## Results
- **Training Accuracy**: Improved from 88.86% to 94.14%
- **Validation Accuracy**: 96.85% to 98.73%
- **Equal Error Rate (EER)**: 0.0060 at threshold 0.8181
- **False Acceptance Rate (FAR) at EER**: 0.0050
- **False Rejection Rate (FRR) at EER**: 0.0070

## Changes Made
1. **Data Loading and Preprocessing**:
   - Loaded synthetic fingerprint data from .npz files
   - Applied data augmentation using imgaug library
   - Split data into training and validation sets

2. **Model Architecture**:
   - Implemented Siamese network with VGG16 base
   - Added custom layers for feature extraction
   - Used contrastive loss for training

3. **Training Pipeline**:
   - Configured training with appropriate batch size and epochs
   - Implemented early stopping and model checkpointing
   - Added visualization of training progress

4. **Evaluation**:
   - Implemented EER calculation
   - Added ROC curve plotting
   - Created matching visualization for sample pairs

5. **Dependencies**:
   - Updated requirements.txt with specific versions
   - Resolved compatibility issues (e.g., numpy version for imgaug)

## Technical Implementation
- **Framework**: TensorFlow/Keras
- **Data Format**: NumPy arrays (.npy, .npz)
- **Visualization**: Matplotlib for plots and image display
- **Preprocessing**: OpenCV and scikit-image for image processing

## Conclusion
The fingerprint matching system achieves high accuracy with an EER of 0.60%, demonstrating effective fingerprint verification capabilities. The Siamese network architecture successfully learns discriminative features for fingerprint matching.