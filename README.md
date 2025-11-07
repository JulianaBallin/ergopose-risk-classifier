# 🧠 ErgoPose Risk Classifier  
*A lightweight neural network-based system for ergonomic risk assessment using webcam posture data.*

---

## 📍 Project Overview

This project aims to develop a **computer vision system** that monitors human posture in real time through a **webcam**, detects **body landmarks (pose estimation)**, and computes key **angles** — neck, trunk, and shoulder. Using a **supervised machine learning model (Artificial Neural Network)**, the system classifies the **postural risk** as *low*, *medium*, or *high* during seated work.

The goal is to contribute to **ergonomic safety** and **occupational health**, offering a fast and accessible approach to identify poor postures that may lead to musculoskeletal disorders.

Dataset reference: [Zenodo - Postural Risk Estimation Dataset (2024)](https://zenodo.org/records/14230872)

---

## 👩‍💻 Team Members

| Name | Registration | GitHub Profile |
|------|---------------|----------------|
| **Juliana Ballin Lima** | 2315310011 | [GitHub Profile](https://github.com/JulianaBallin) |
| **Marcelo Heitor de Almeida Lira** | 2315310043 | [GitHub Profile](https://github.com/Marcelo-Heitor-de-Almeida-Lira) |
| **Lucas Maciel Gomes** | 2315310014 | [GitHub Profile](https://github.com/lucassmaciel) |
| **Ryan da Silva Marinho** | 2315310047 | [GitHub Profile](https://github.com/RyanDaSilvaMarinho) |

---

## 📂 Repository Structure

```
ergopose-risk-classifier/
│
├── data/
│ ├── raw/ # Original dataset files from Zenodo
│ └── processed/ # Cleaned and transformed data
│
├── notebooks/
│ ├── 01_data_preparation.ipynb
│ ├── 02_exploratory_analysis.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_results_analysis.ipynb
│
├── models/
│ ├── neural_network.pkl # Trained model
│ └── scaler.pkl # Feature scaler for input normalization
│
├── src/
│ ├── data_preprocessing.py # Data cleaning and transformation functions
│ ├── feature_extraction.py # Body angle computation from landmarks
│ ├── model_training.py # ANN definition, training, and evaluation
│ └── inference.py # Real-time posture classification using webcam
│
├── slides/
│ └── presentation.pptx # Final presentation
│
├── requirements.txt # Python dependencies
├── README.md
└── LICENSE
```

---

## ⚙️ Methodology and Steps

### 1. **Data Collection & Preparation**
We use an open dataset containing annotated posture keypoints and ergonomic risk levels.  
Preprocessing steps include:
- Normalization and standardization of coordinates.
- Calculation of angles (neck, trunk, shoulders) from body landmarks.
- Label encoding for risk classification (0 = low, 1 = medium, 2 = high).

### 2. **Exploratory Data Analysis (EDA)**
- Visualization of body joint distributions and angle ranges.
- Correlation between features and ergonomic risk.
- Detection of outliers and class balance verification.

### 3. **Learning Task**
Supervised classification using an **Artificial Neural Network (ANN)**.  
The task aims to predict the ergonomic risk level from a set of calculated angles.

### 4. **Validation Strategy**
- **k-Fold Cross-Validation (k=5)** to ensure generalization.
- Evaluation metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- Comparison with baseline models (e.g., Decision Tree, SVM).

### 5. **Model and Hyperparameter Grid Search**
The following hyperparameters were tuned:
- Hidden layers: [1, 2, 3]
- Neurons per layer: [8, 16, 32, 64]
- Activation functions: [ReLU, Tanh]
- Optimizers: [Adam, SGD]
- Learning rates: [0.001, 0.01, 0.1]
- Batch sizes: [8, 16, 32]

### 6. **Training and Testing**
Training and evaluation were carried out using **TensorFlow/Keras**.  
Training curves, loss evolution, and confusion matrices are included in the notebooks.

### 7. **Results and Analysis**
- Visual comparison of prediction accuracy across models.
- Discussion of misclassification cases.
- Proposal for real-time inference using **OpenCV + MediaPipe** integration.

---

## 📊 Expected Outputs
- Trained ANN model and performance metrics.  
- Visualization of key performance indicators (loss, accuracy, confusion matrix).  
- Real-time webcam demonstration classifying posture risk levels.

---

## 🧩 Tools and Technologies
| Category | Tools |
|-----------|--------|
| Programming Language | Python 3.11 |
| Machine Learning | TensorFlow / Keras, Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Computer Vision | OpenCV, MediaPipe |
| Development | Jupyter Notebook, GitHub |
| Dataset | [Zenodo Postural Risk Dataset (2024)](https://zenodo.org/records/14230872) |

---

## 🎯 Expected Learning Outcomes
- Understanding of the **supervised learning pipeline**.
- Application of **Artificial Neural Networks** for ergonomic risk classification.
- Integration of **pose estimation** with machine learning.
- Development of research, collaboration, and documentation skills.

---

## 🧾 License
This project is licensed under the [MIT License](LICENSE).

---

## 🧑‍🏫 Academic Context
This repository was developed as the **Final Evaluative Activity (AA3)** for the course *Machine Learning* (Universidade do Estado do Amazonas, 2025).  
It includes all required stages: data preparation, exploratory analysis, task definition, cross-validation, hyperparameter search, model training, performance analysis, and presentation slides.

---

*“Artificial Intelligence supporting healthy workplaces — one posture at a time.”*
