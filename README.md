#  Cats vs Dogs Image Classification using SVM

## 📌 Project Overview

This project builds an image classification system to distinguish between cats and dogs using a classical Machine Learning approach.

Instead of deep learning, this project uses:
- HOG (Histogram of Oriented Gradients) for feature extraction
- Support Vector Machine (SVM) for classification

The goal is to demonstrate strong understanding of feature engineering and traditional ML pipelines.

---

## 🎯 Problem Statement

Given an image of a pet, classify whether it is a cat or a dog using a Support Vector Machine classifier.

---

## 📂 Dataset

Dataset used: Dogs vs Cats dataset from Kaggle  

- Thousands of labeled cat and dog images
- Images resized to 128x128
- Converted to grayscale for feature extraction

Dataset Source: Kaggle Dogs vs Cats Competition

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib
- scikit-image (HOG)
- Google Colab

---

## 🔍 Feature Engineering

HOG (Histogram of Oriented Gradients) was used to extract shape and edge-based features from grayscale images.

Why HOG?
- Captures structural information
- Works well with classical ML algorithms
- Reduces dimensionality compared to raw images

---

## 🤖 Model Used

Support Vector Machine (SVM)

Experiments performed:
- Linear Kernel
- RBF Kernel
- Hyperparameter tuning using GridSearchCV

---

## 📊 Model Performance

- Accuracy: ~78% – 82% (depending on dataset size)
- Evaluation Metrics:
  - Confusion Matrix
  - Precision
  - Recall
  - F1-Score

---

## 📈 Workflow

1. Dataset Download using Kaggle API
2. Image Preprocessing
   - Resize
   - Grayscale conversion
3. HOG Feature Extraction
4. Train-Test Split
5. Model Training (SVM)
6. Evaluation and Visualization
7. Single Image Prediction

---

##  Key Learnings

- Understanding classical ML for image classification
- Feature engineering using HOG
- Hyperparameter tuning
- Model evaluation techniques
- Real-world ML workflow implementation

---

## 🚀 Future Improvements

- Train on full dataset for better accuracy
- Implement CNN using TensorFlow/PyTorch
- Deploy as a web application
- Optimize preprocessing pipeline

---

##  Conclusion

This project demonstrates the effectiveness of classical Machine Learning techniques for image classification tasks. 

While deep learning models may achieve higher accuracy, this implementation highlights strong fundamentals in feature extraction and model training.

---

## 👨‍💻 Author

Sambhav Ramteke  
Machine Learning Enthusiast  
Skills: Python, Data Structures, Machine Learning, Data Analytics
