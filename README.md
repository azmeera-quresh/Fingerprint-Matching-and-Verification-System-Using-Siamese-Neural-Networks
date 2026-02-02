# Fingerprint Matching and Verification System

A comprehensive fingerprint recognition system using deep learning (Siamese neural networks) combined with classical fingerprint preprocessing techniques.

## Project Overview

This project implements an end-to-end fingerprint verification pipeline with two main components:

1. **Preprocessing Notebook** - Feature extraction and analysis
2. **Recognition Notebook** - Siamese network training and evaluation

## Dataset

- **SOCOFing Real** (6,000 images): High-quality fingerprint images from live scanners
  - Used for: Feature learning and preprocessing analysis
  - Format: .BMP files (480×640 pixels)
  
- **SOCOFing Altered** (49,270 images): Synthetic variations with difficulty levels
  - Used for: Training and validation
  - Format: .NPY arrays (90×90 pixels)
  - Categories: Easy, Medium, Hard

## Pipeline Architecture

### 1. Preprocessing (`Preproceesing in fingerprint matching.ipynb`)
- Image loading and visualization
- Noise reduction (Gaussian & median blur)
- Binarization (mean-based & Otsu thresholding)
- Edge detection (Robert, Sobel, Prewitt filters)
- Ridge detection (Hessian eigenvalue analysis)
- Skeletonization
- **Minutiae extraction** (termination & bifurcation points)
- Orientation computation

### 2. Recognition (`Fingerprint Recognition .ipynb`)
- Data loading from .NPY format
- Data augmentation (rotation, scale, translation, blur)
- Image enhancement (histogram equalization + Gabor filtering)
- **Siamese network** architecture:
  - Shared convolutional feature extractor
  - Subtraction layer for comparison
  - Binary classifier (matched vs. unmatched)
- Training with binary crossentropy loss
- Evaluation metrics: ROC curve, AUC, FAR, FRR, **EER**

## Key Features

✅ **Full Preprocessing Pipeline**: From raw fingerprints to extracted minutiae  
✅ **Siamese Architecture**: Learns fingerprint matching via paired image comparison  
✅ **Multi-difficulty Dataset**: Easy/Medium/Hard samples for robust training  
✅ **Data Augmentation**: Realistic variations to prevent overfitting  
✅ **Comprehensive Evaluation**: ROC/EER/FAR/FRR metrics for biometric systems  

## Technologies Used

- **Python 3.x**
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, scikit-image
- **Data Augmentation**: imgaug
- **Analysis**: NumPy, scikit-learn, Matplotlib, Seaborn

## Methodology

### Fingerprint Image Processing
Fingerprint images are standardized to eliminate inconsistent clarity, grayscale, and channel variations:

1. **Gray Scaling**: Convert to grayscale to focus on ridge intensity information
2. **Histogram Equalization**: Enhance contrast and normalize grayscale values
3. **Noise Reduction**: Apply Gaussian/median filtering to smooth images
4. **Binarization**: Use mean-based or Otsu thresholding for segmentation
5. **Edge Detection**: Compare Robert, Sobel, Prewitt filters
6. **Ridge Detection**: Use Hessian eigenvalue analysis for ridge maps
7. **Skeletonization**: Thin ridges to 1-pixel width
8. **Minutiae Extraction**: Detect termination and bifurcation points with orientations

### Siamese Network Architecture

The Siamese network learns fingerprint matching via paired image comparison:

```
Input Layer (Two paired fingerprints)
    ↓
Shared Feature Extractor (Conv2D blocks + MaxPool)
    ↓
[Feature Map 1] and [Feature Map 2]
    ↓
Subtract Layer (element-wise difference)
    ↓
Classification Head (Conv2D → Flatten → Dense → Sigmoid)
    ↓
Output: Similarity score (0=different, 1=same fingerprint)
```

**Loss Function**: Binary Crossentropy  
**Optimizer**: Adam  
**Metrics**: Accuracy, ROC-AUC, EER

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
a total of six images. Through this data augmentation technique, the number of images for
each fingerprint increased to an average of 60. Regarding the training of the comparative
network, two images of the same fingerprint kind were selected from the training set, and
the network output was calibrated to 1. Similarly, an image of a different fingerprint kind
was selected, and the output was calibrated to 0. This process was repeated with different
images from the training set. By training the network in this manner, when two fingerprint
images of the same finger were input, the network output was biased towards 1. Conversely,
when two fingerprint images of different fingers were input, the network output was biased
towards 0. 


![model](https://github.com/Nitheshkamath/Fingerprint_Matching/assets/112107488/1613b2c9-f969-45a1-842f-c2c579868888)

# Results and Evaluation

## Project Team
- **Azmeera Qureshi** (2103626)
- **Hamza Munzir** (2243239)

## Siamese Network Performance
The fingerprint matching system was successfully implemented using a Siamese neural network architecture with VGG16 pre-trained weights. The model achieved excellent performance on synthetic fingerprint data:

- **Validation Accuracy**: 98.73%
- **Equal Error Rate (EER)**: 0.60%
- **Training Epochs**: 50 epochs with early stopping
- **Architecture**: Siamese network with contrastive loss function

The ROC curve analysis showed strong discriminative power, with the model effectively distinguishing between genuine and impostor fingerprint pairs.

## Fingerprint Preprocessing Pipeline
The preprocessing pipeline was successfully demonstrated on real SOCOFing fingerprint dataset, implementing comprehensive image processing techniques:

### 1. Data Loading and Visualization
- Successfully loaded 6000 real fingerprint images from SOCOFing dataset
- Displayed sample fingerprint images for visual inspection

### 2. Image Enhancement Techniques
- **Gaussian Blurring**: Applied with 1x1 kernel for noise reduction
- **Median Blurring**: Applied with 3x3 kernel for salt-and-pepper noise removal
- **Histogram Analysis**: Generated intensity distribution plots for image characterization

### 3. Thresholding Methods
- **Mean Thresholding**: Applied dynamic thresholds based on image mean values
- **Otsu's Thresholding**: Automatic threshold selection for optimal binarization

### 4. Edge Detection Filters
Successfully implemented and compared three edge detection techniques:
- **Robert Filter**: Simple 2x2 gradient operator
- **Sobel Filter**: 3x3 gradient operator with better noise suppression
- **Prewitt Filter**: 3x3 gradient operator similar to Sobel

### 5. Ridge Detection
- **Hessian Matrix Analysis**: Used for ridge/valley detection
- **Sigma Parameter**: Optimized at 0.15 for fingerprint ridge extraction

### 6. Minutiae Extraction
Successfully extracted fingerprint minutiae points:
- **Termination Points**: Ridge endings (marked in blue)
- **Bifurcation Points**: Ridge branchings (marked in red)
- **Skeletonization**: Applied morphological operations for ridge thinning
- **Feature Classification**: Automatic detection and labeling of minutiae types

## Technical Implementation Details
- **Programming Language**: Python 3.10.0
- **Deep Learning Framework**: TensorFlow/Keras with VGG16 backbone
- **Image Processing**: OpenCV, scikit-image, PIL
- **Data Augmentation**: Rotation-based augmentation (5 angles per image)
- **Loss Function**: Contrastive loss for Siamese network training
- **Dataset**: SOCOFing real fingerprint database (6000 images)

## Key Achievements
1. **High Accuracy Matching**: Achieved 98.73% validation accuracy on fingerprint verification task
2. **Low Error Rate**: Maintained 0.60% EER, indicating robust performance
3. **Complete Preprocessing Pipeline**: Successfully implemented all major fingerprint preprocessing techniques
4. **Real Data Validation**: Demonstrated preprocessing on real-world SOCOFing dataset
5. **Minutiae-Based Features**: Extracted discriminative fingerprint features for matching

## Future Improvements
- Integration of preprocessing pipeline with matching system
- Testing on larger and more diverse fingerprint datasets
- Implementation of additional enhancement techniques (Gabor filtering, orientation estimation)
- Deployment as a complete fingerprint recognition system

---

*This project demonstrates a comprehensive approach to fingerprint recognition, combining deep learning-based matching with traditional image processing techniques for robust biometric identification.*