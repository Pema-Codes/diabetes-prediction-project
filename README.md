# Diabetes prediction 
## Project Overview

This project develops a high-performance machine learning pipeline to predict diabetes in patients based on clinical features. Unlike generic models, this project specifically prioritizes Patient Safety by optimizing the model to minimize False Negatives (missed diagnoses).

## Performance Overview: 

Highest ROC AUC: 0.9762 (on unseen test data).

Cross-Validation Stability: 0.9973 (indicating a robust, non-overfit model).

Recall (Class 1)	0.79 (79% of diabetic cases caught (Optimized)).

Clinical Optimization: Reduced False Negatives by 26% through strategic threshold tuning (False Negatives-354)

 ## Tech Stack

 Language : Python 3.x 

 Core Libraries: Pandas, NumPy (Data Manipulation)

 Visualization: Matplotlib, Seaborn (EDA & Insights)

 Machine Learning: Sci-kit Learn (Preprocessing/ Evaluation)

 Gradient Boosting: XGBoost (Leading Algorithm)

 Sampling: Imbalanced-Learn (SMOTE)

### Project Milestones

Exploratory Data Analysis (EDA): Identified Glucose and HbA1c as primary predictors via correlation mapping. 

Preprocessing: Cleaned categorical inconsistencies and implemented One-Hot Encoding for demographic data. 

Imbalance Correction: Utilized SMOTE to balance the training set (91/9 split to 50/50).

Model Selection: Benchmarked Logistic Regression, Decision Trees, and Random Forest against XGBoost. 

Hyperparameter Optimization: Conducted GridSearchCV to fine-tuning learning rates and tree depth. 

Threshold Engineering: Adjusted decision boundaries to 0.3 for clinical safety. 

## Technical Deep-Dive
### The "Safe" Threshold Strategy

Standard models use a 0.5 threshold. In healthcare, a False Negative (missing a sick patient) is far more costly than a False Positive (a false alarm). By shifting the threshold to 0.3, I forced the model to be more "suspicious," significantly increasing the Recall of the diabetic class.

### Hyperparameter Logic

I used GridSearchCV to solve for Variance (Overfitting).

   -  max_depth: 6: Constrained the trees to prevent the model from memorizing noise.

   - learning_rate: 0.1: Ensured the model converged slowly and accurately.
     
## Visual Insights
### Feature Importance (Gain)

The model identified that HbA1c_level and blood_glucose_level contribute the most to the diagnostic decision. This aligns with clinical standards, validating that the model learned medical logic, not just random correlations.

### Confusion Matrix Analysis

The matrix reveals the trade-off made for safety. By accepting slightly more False Positives (top-right), we successfully pushed our False Negatives (bottom-left) to a lower, safer level for medical screening.

## Model Performance Summary

XGBoost (Tuned)-  Accuracy: 95.14%, ROC: 0.9762

Random Forest- Accuracy: 95.87%, ROC: 0.9653

Logistic Regression- Accuracy: 88.66%, ROC: 0.9604

## How to Run

Clone the repository: git clone https://github.com/Pema-Codes/diabetes-prediction.git

Install requirements: pipp install pandas scikit-learn xgboost matplotlib seaborn imabalanced-learn

Run the notebook: Open Diabetes_prediciton.ipynb in Google Colab or Jupyter. 

Load the model: 

import joblib 

model = joblib.load('best_xgboost_model.pkl')
