# Breast Cancer Classification with Classical Machine Learning

This project demonstrates a complete classical machine learning workflow using a structured numerical dataset.
The focus is on **data preprocessing, dimensionality reduction, classification, and clustering**, without neural networks.

## Project Goals

- Predict a binary target variable
- Apply Support Vector Machine (SVM)
- Use Principal Component Analysis (PCA)
- Explore clustering in reduced-dimensional space

## How to run the project

This project is implemented as a Jupyter Notebook.

- Open `breast_cancer_analysis.ipynb`

To run the notebook locally, install the following Python packages: <br>
```pip install pandas numpy matplotlib scikit-learn jupyter```

## Dataset Overview

- Dataset source: [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)
- Format: CSV
- Rows: 569
- Features: 30 numerical measurements
- Target:
  - 1 = malignant
  - 0 = benign

## Methods Used

### Data Preprocessing
- Removed non-informative ID column
- Encoded target variable
- Train–test split with stratification
- Feature standardization

### Classification
- Support Vector Machine (SVM)
- Accuracy-based evaluation

### Dimensionality Reduction
- PCA with variance analysis
- Reduced feature space to 10 components

### Clustering
- KMeans clustering in PCA space
- Silhouette score evaluation

## Results
| Model | Accuracy |
|------|----------|
| SVM (full features) | ~96.5% to 98.2% |
| SVM (PCA-reduced) | ~95.6% to 97.4% |

## Key Observations

- Feature scaling is essential
- PCA preserves most relevant information
- Reduced dimensionality gives similar performance
- Clustering reveals overlapping class structure

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Project Structure
├── breast-cancer.csv <br>
├── breast_cancer_analysis.ipynb <br>
├── README.md <br>
