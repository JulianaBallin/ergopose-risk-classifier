# ErgoPose Risk Classifier  
*Binary Neural Network for 2D Ergonomic Posture Quality Assessment (Good vs Bad).*

---

## 📍 Project Overview

This project extends the original MultiPosture dataset experiment by introducing data simplification and generalization challenges — removing the Z coordinate, excluding subject identifiers, and incorporating a custom Quality Index to evaluate posture stability.

The goal is to contribute to **ergonomic safety** and **occupational health**, offering a fast and accessible approach to identify poor postures that may lead to musculoskeletal disorders.

Dataset reference: [Zenodo - Postural Risk Estimation Dataset (2024)](https://zenodo.org/records/14230872)

---

## 👩‍💻 Team Members

| Name | Registration | GitHub Profile |
|------|--------------|----------------|
| **Juliana Ballin Lima** | 2315310011   | [GitHub Profile](https://github.com/JulianaBallin) |
| **Marcelo Heitor de Almeida Lira** | 2315310043   | [GitHub Profile](https://github.com/Marcelo-Heitor-de-Almeida-Lira) |
| **Lucas Maciel Gomes** | 2315310014   | [GitHub Profile](https://github.com/lucassmaciel) |
| **Ryan da Silva Marinho** | 2315310047   | [GitHub Profile](https://github.com/RyanDaSilvaMarinho) |
| **Pedro César Mendonça Ituassú** | 2315080063   | [GitHub Profile](https://github.com/pedroituassu) |
| **Caio Jorge Da Cunha Queiroz** | 2315310028   | [GitHub Profile](https://github.com/cjcaio) |

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
│ └── 04_fine_tune_models.ipynb
│ └── 05_results_analysis.ipynb
│
├── models/
│ ├── neural_network.pkl # Trained model
│ └── scaler.pkl # Feature scaler for input normalization
│
├── documents/
│ └── presentation.pdf
│ └── diagram-pipeline.json
│ └── diagram-pipeline.png 
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
- Label **binarization for Good vs Bad posture quality** classification (good = TUP, bad = all other upper-body labels).

### 2. **Feature Selection**
Feature relevance is evaluated through:
- Pearson correlation analysis;
- `SelectKBest` and/or `Recursive Feature Elimination (RFE)`;
- Manual validation using domain knowledge (ergonomic criteria).

Irrelevant or redundant features are dropped to improve model performance and interpretability.

### 3. **Learning Task**
A **binary supervised classification task** using an Artificial Neural Network (ANN).  
The ANN predicts **posture quality (Good vs Bad)** with a single sigmoid output neuron.

### 4. **Validation Strategy**
- **5-Fold Cross-Validation** to evaluate model robustness.
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix** for the two classes (Good vs Bad).
- Comparison with baseline algorithms (Decision Tree, SVM).

### 5. **Model Architecture and Hyperparameter Tuning**
In compliance with the project constraints, we will explore architectures where:
- **Total hidden neurons**: Between 5 and 20.
- **Hidden layers**: If more than one layer, the sum of their neurons must be between 5 and 20.
- **Activation functions**: ReLU, Sigmoid, etc. (excluding Tanh).
- **Learning rates**: 0.01, 0.001, or smaller.
- **Batch size**: Default Keras value (32).

Our search will focus on finding the optimal combination within these boundaries to maximize performance on the training set.

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
**DATA COLLECTION**
──────────────────────────────────────────────────────────────  
• Dataset: MultiPosture (Zenodo, 2024)  
• 13 participants — 4,800 frames — 11 joints (x, y, z)  
• Labels: upper and lower body posture classes  


──────────────────────────────────────────────────────────────  
**DATA PREPARATION & CLEANING**
──────────────────────────────────────────────────────────────  
• Remove Z coordinates → 2D-only input  
• Remove subject ID → ensure model generalization  
• Normalize and standardize coordinates  
• Compute body angles (neck, trunk, shoulder)  
• Create "Quality Index" → stability metric based on angle variation  


──────────────────────────────────────────────────────────────  
**FEATURE SELECTION**
──────────────────────────────────────────────────────────────  
• Pearson correlation analysis  
• SelectKBest or Recursive Feature Elimination (RFE)  
• Manual validation using ergonomic domain knowledge  


──────────────────────────────────────────────────────────────  
**MODEL TRAINING**
──────────────────────────────────────────────────────────────  
• Artificial Neural Network (ANN) for binary classification  
• Input: selected features + quality index  
• Output: posture quality (0 = Bad, 1 = Good)
• Framework: TensorFlow / Keras  

     
──────────────────────────────────────────────────────────────  
**CROSS-VALIDATION**
──────────────────────────────────────────────────────────────  
• 5-Fold cross-validation  
• Metrics: Accuracy, Precision, Recall, F1-Score  
• Baseline comparison: SVM, Decision Tree  


──────────────────────────────────────────────────────────────  
**RESULTS ANALYSIS**
──────────────────────────────────────────────────────────────  
• Compare performance with / without feature selection  
• Confusion matrix and misclassification analysis  
• Evaluate 2D (no Z) vs 3D models  
• Visualize loss and accuracy curves  

──────────────────────────────────────────────────────────────  

## 📊 Expected Outputs
- **Trained ANN models** for binary posture quality classification (saved as `.h5` or `.pkl`).
- **Training history logs** for all model variants.
- **Comparative performance histograms** generated from the history, providing a final overview of model accuracy.
- Comparative performance metrics (2D vs 3D, with/without feature selection).
- Visualizations: learning curves, confusion matrices, and feature importance plots.
- Real-time webcam demo classifying posture as Good or Bad.  

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
- Build and evaluate binary neural networks for posture quality recognition.
- Develop critical thinking about model generalization and data bias.
- Integrate ergonomic domain knowledge into ML workflows.

---

## 🧑‍🏫 Academic Context
Developed as the **Final Project (AA3)** for the course *Neural Networks and Deep Learning* — Universidade do Estado do Amazonas (2025).  
Includes all required stages: preprocessing, feature selection, model training, evaluation, and presentation.

---

*“Artificial Intelligence supporting healthy workplaces — one posture at a time.”*

