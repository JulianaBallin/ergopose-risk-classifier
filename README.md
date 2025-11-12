# 🧠 ErgoPose Risk Classifier  
*Multi-Class Neural Network for 2D Ergonomic Posture Recognition and Risk Assessment.*

---

## 📍 Project Overview

This project extends the original MultiPosture dataset experiment by introducing data simplification and generalization challenges — removing the Z coordinate, excluding subject identifiers, and incorporating a custom Quality Index to evaluate posture stability.

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
| **Pedro César Mendonça Ituassú** | 2315080063 | [GitHub Profile](https://github.com/pedroituassu) |
| **Caio Jorge Da Cunha Queiroz** | 2315310047 | [GitHub Profile](https://github.com/cjcaio) |

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

The methodology was adapted to introduce additional experimental constraints for model robustness and to promote deeper understanding of feature relevance and generalization.

### 1. **Data Collection & Preparation**
We use the **MultiPosture Dataset** (Zenodo, 2024), containing skeletal pose keypoints extracted via MediaPipe.  
Preprocessing steps include:
- Removal of **Z coordinates** to simulate 2D-only analysis.
- Removal of **subject ID** to ensure generalization across individuals.
- **Feature engineering**: computation of neck, trunk, and shoulder angles.
- **Quality index creation** — a stability metric based on the variation of body angles.
- Normalization and standardization of all numerical features.
- Label encoding for **multi-class posture classification** (e.g., TUP, TLF, TLB, etc.).

### 2. **Feature Selection**
Feature relevance is evaluated through:
- Pearson correlation analysis;
- `SelectKBest` and/or `Recursive Feature Elimination (RFE)`;
- Manual validation using domain knowledge (ergonomic criteria).

Irrelevant or redundant features are dropped to improve model performance and interpretability.

### 3. **Learning Task**
A **multi-class supervised classification** task using an **Artificial Neural Network (ANN)**.  
The ANN predicts the **upper-body posture class** based on computed features and the stability index.

### 4. **Validation Strategy**
- **5-Fold Cross-Validation** to evaluate model robustness.
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix** for each class.
- Comparison with baseline algorithms (Decision Tree, SVM).

### 5. **Model and Hyperparameter Grid Search**
Tuning parameters include:
- Hidden layers: [1, 2, 3]
- Neurons per layer: [8, 16, 32, 64]
- Activation functions: [ReLU, Tanh]
- Optimizers: [Adam, SGD]
- Learning rates: [0.001, 0.01, 0.1]
- Batch sizes: [8, 16, 32]

### 6. **Training and Testing**
Training and evaluation conducted with **TensorFlow/Keras**.  
Notebooks include:
- Learning curves and loss analysis.
- Comparison between original 3D vs 2D (no Z) models.
- Evaluation of feature selection impact.

### 7. **Results and Analysis**
- Accuracy comparison with and without feature selection.
- Influence of the stability index on classification.
- Error analysis and confusion matrices.
- Insights on how the ANN generalizes across participants.
- Evaluation of class-wise precision and recall, identifying which postures are most difficult to classify.
- Analysis of the impact of removing the Z coordinate on spatial feature learning.

---
## 🧠 Machine Learning Pipeline – Ergonomic Posture Classifier

──────────────────────────────────────────────────────────────
📥 DATA COLLECTION
──────────────────────────────────────────────────────────────
• Dataset: MultiPosture (Zenodo, 2024)
• 13 participants — 4,800 frames — 11 joints (x, y, z)
• Labels: upper and lower body posture classes


──────────────────────────────────────────────────────────────
🧹 DATA PREPARATION & CLEANING
──────────────────────────────────────────────────────────────
• Remove Z coordinates → 2D-only input  
• Remove subject ID → ensure model generalization  
• Normalize and standardize coordinates  
• Compute body angles (neck, trunk, shoulder)  
• Create "Quality Index" → stability metric based on angle variation  


──────────────────────────────────────────────────────────────
🔍 FEATURE SELECTION
──────────────────────────────────────────────────────────────
• Pearson correlation analysis  
• SelectKBest or Recursive Feature Elimination (RFE)  
• Manual validation using ergonomic domain knowledge  


──────────────────────────────────────────────────────────────
🧠 MODEL TRAINING
──────────────────────────────────────────────────────────────
• Artificial Neural Network (ANN) for multi-class classification  
• Input: selected features + quality index  
• Output: posture classes (TUP, TLF, TLB, etc.)  
• Framework: TensorFlow / Keras  

     
──────────────────────────────────────────────────────────────
🧪 CROSS-VALIDATION
──────────────────────────────────────────────────────────────
• 5-Fold cross-validation  
• Metrics: Accuracy, Precision, Recall, F1-Score  
• Baseline comparison: SVM, Decision Tree  


──────────────────────────────────────────────────────────────
📊 RESULTS ANALYSIS
──────────────────────────────────────────────────────────────
• Compare performance with / without feature selection  
• Confusion matrix and misclassification analysis  
• Evaluate 2D (no Z) vs 3D models  
• Visualize loss and accuracy curves  


──────────────────────────────────────────────────────────────
🤖 REAL-TIME INFERENCE (DEMO)
──────────────────────────────────────────────────────────────
• Integration with OpenCV + MediaPipe  
• Webcam-based posture risk classification  
• Real-time ergonomic feedback: Low / Medium / High  
──────────────────────────────────────────────────────────────

---



## 📊 Expected Outputs
- Trained ANN for multi-class posture classification.
- Comparative performance metrics (2D vs 3D, with/without feature selection).
- Visualizations: learning curves, confusion matrices, and feature importance plots.
- Real-time webcam demo classifying posture classes.

---

## 🧩 Tools and Technologies
| Category | Tools |
|-----------|--------|
| Language | Python 3.11 |
| ML Frameworks | TensorFlow, Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Computer Vision | MediaPipe, OpenCV |
| Development | Jupyter Notebook, GitHub |
| Dataset | [Zenodo MultiPosture Dataset (2024)](https://zenodo.org/records/14230872) |

---

## 🎯 Expected Learning Outcomes
- Apply **feature engineering and selection** in supervised learning.
- Understand the trade-offs of **data simplification (Z removal)**.
- Build and evaluate **multi-class neural networks**.
- Develop critical thinking about model generalization and data bias.
- Integrate ergonomic domain knowledge into ML workflows.

---

## 🧑‍🏫 Academic Context
Developed as the **Final Project (AA3)** for the course *Neural Networks and Deep Learning* — Universidade do Estado do Amazonas (2025).  
Includes all required stages: preprocessing, feature selection, model training, evaluation, and presentation.

---

*“Artificial Intelligence supporting healthy workplaces — one posture at a time.”*

