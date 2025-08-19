# Diabetes Prediction - Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning project for predicting diabetes based on patient clinical data. This project explores multiple classification algorithms and provides detailed analysis from data preprocessing to model evaluation.

## 📋 Project Overview

This project uses the **Pima Indians Diabetes Database** from Kaggle to develop predictive models for diabetes diagnosis. The dataset contains medical diagnostic measurements and the goal is to predict whether a patient has diabetes based on these features.

## 🚀 Features

- **Exploratory Data Analysis (EDA)** with detailed visualizations
- **Advanced preprocessing** handling missing values, outliers, and feature scaling
- **Class imbalance handling** using SMOTE technique
- **Multiple ML models** comparison:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (Neural Network)
- **Comprehensive evaluation** with cross-validation and performance metrics
- **PCA analysis** for dimensionality reduction

## 📊 Dataset Features

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0: non-diabetic, 1: diabetic)

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 84.25% | 84.69% | 84.25% | 84.18% | 0.9537 |
| Random Forest | 88.19% | 88.58% | 88.19% | 88.15% | 0.9657 |
| SVM | 84.40% | 84.73% | 84.42% | 84.36% | 0.9544 |
| **MLP (Neural Network)** | **90.48%** | **90.39%** | **90.52%** | **90.45%** | **0.9717** |

## 🏆 Best Performing Model

The **Multi-Layer Perceptron (MLP)** neural network achieved the best performance with:
- **Accuracy**: 90.48%
- **AUC-ROC**: 0.9717
- **Excellent class separation capability**

## 🛠️ Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/SinghProbjot/diabetes-prediction-ml.git
cd diabetes-prediction-ml

pip install -r requirements.txt

jupyter notebook diabetes_prediction_analysis.ipynb
