# Explainable Road Risk Prediction using Computer Vision & XAI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

This project presents a **context-aware road risk prediction system** that classifies driving scenes into **Low, Medium, and High risk** categories using dashcam images.

Instead of directly using raw images, the system extracts **contextual traffic features** using object detection and combines **unsupervised learning, supervised modeling, and Explainable AI (XAI)** for interpretable risk prediction.

---

##  Key Features

- YOLO-based object detection for scene understanding  
- Contextual feature extraction (traffic density, proximity, etc.)  
- KMeans clustering for **pseudo-label generation (weak supervision)**  
- Multi-Layer Perceptron (MLP) for classification  
- SHAP-based Explainable AI for model interpretability  

---

##  Project Pipeline
Dashcam Image
↓
YOLO Object Detection
↓
Feature Extraction
↓
KMeans Clustering (Label Generation)
↓
MLP Classifier (Risk Prediction)
↓
SHAP (Explainability)

---

## 📂 Project Structure
├── 1_risk_level_yolo_perception.ipynb
├── 2_clustering_for_label_generation.ipynb
├── 3_risk_classifier_final.ipynb
├── extracted_features_with_risk_labels.csv
└── README.md

---

## 📊 Dataset

The project uses the **BDD100K dataset**:

🔗 https://www.kaggle.com/datasets/aayusmaanjain/bdd100k-for-self-driving-cars

- Real-world dashcam images  
- Diverse lighting and weather conditions  
- Annotated with object bounding boxes  

---

## ⚙️ Methodology

### 1. Perception (YOLO)
- Detect vehicles, pedestrians, and road objects  
- Extract bounding box information  

### 2. Feature Engineering
- Vehicle count  
- Pedestrian count  
- Object density  
- Average bounding box area  
- Maximum bounding box area  

### 3. Risk Label Generation (KMeans)
- Unsupervised clustering (k = 3)  
- Generates Low / Medium / High risk labels  
- Validated using Elbow Method and Silhouette Score  

### 4. Risk Classification (MLP)
- Input: 5 contextual features  
- Optimized neural network architecture  
- Loss: CrossEntropy  
- Optimizer: Adam  

### 5. Explainability (SHAP)
- Provides feature-level interpretation  
- Explains contribution of each feature to predictions  

---

## 📈 Results

- **Test Accuracy:** ~93.8%  
- Strong performance for Low and High risk classes  
- Moderate performance for Medium risk due to overlap  

---

## 📌 Key Insights

- Contextual features effectively capture traffic risk  
- Traffic density and proximity are major indicators  
- Weak supervision eliminates need for manual labeling  
- Simple models perform well due to strong feature separability  

---

## ⚠️ Limitations

- Medium risk class shows overlap with other categories  
- No temporal (video) information considered  
- Performance depends on object detection quality  

---

## 🚀 Future Work

- Incorporate spatial relationships between objects  
- Extend to video-based analysis  
- Improve class imbalance handling  
- Deploy in real-time environments  

---

## 🛠️ Requirements
Python
PyTorch
scikit-learn
OpenCV
NumPy
Pandas
Matplotlib / Seaborn
SHAP
---
## ⚠️ Important Note

- Update all **file and dataset paths** in the notebooks according to your local environment.
- The current paths may be configured for Google Colab or a specific directory structure.

Example:

```python
img_path = "/content/nn_dataset/Data/train/..."

---

## ▶️ How to Run

1. Clone the repository  
2. Download the dataset from Kaggle  
3. Run notebooks in order:
1_risk_level_yolo_perception.ipynb
→ 2_clustering_for_label_generation.ipynb
→ 3_risk_classifier_final.ipynb

