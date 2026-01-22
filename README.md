# 🏠 Bengaluru House Price Prediction (Machine Learning Project)

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to predict house prices in Bengaluru using real estate data.  
It covers the full ML lifecycle — data cleaning, feature engineering, model training, evaluation, and inference using saved artifacts.

The final selected model is a **Random Forest Regressor**, chosen based on cross-validation performance.

---

## 🎯 Problem Statement
Predict the **market price of a house in Bengaluru** based on features such as:
- Location
- Total square feet
- Number of bedrooms (BHK)
- Number of bathrooms

---

## 🗂 Dataset
- File: `Bengaluru_House_Data2.csv`
- Target Variable: `price` (converted from lakhs to actual value)
- Categorical Feature: `location`
- Numerical Features:
  - `total_sqft`
  - `size` (BHK)
  - `bath`

---

## ⚙️ Data Preprocessing & Feature Engineering

### ✔ Data Cleaning
- Converted `total_sqft` ranges (e.g., `1200-1500`) into numeric averages
- Converted `size` from string format (`"2 BHK"`) to integer
- Removed invalid and missing values

### ✔ Feature Engineering
- Created `price_per_sqft` for outlier detection
- Removed unrealistic price-per-sqft values using domain thresholds
- Removed BHK-based anomalies within the same location

---

## 🔄 Stratified Train-Test Split
- Created a `price_cat` feature to maintain balanced price distribution
- Applied `StratifiedShuffleSplit` to avoid sampling bias

---

## 🧪 Outlier Removal Techniques
- **Price per square foot filtering** using domain-driven thresholds
- **BHK-based outlier removal**
  - Higher BHK houses priced lower than smaller BHKs in the same location were removed

---

## 🔄 Machine Learning Pipeline
Used `ColumnTransformer` to ensure consistent preprocessing.

### Numerical Pipeline
- Median Imputation
- Standard Scaling

### Categorical Pipeline
- One-Hot Encoding (`location`)
- Handles unseen categories safely during inference

---

## 🤖 Models Evaluated

| Model | Mean CV RMSE |
|------|-------------|
| Linear Regression | ~8.46M |
| Decision Tree | ~9.18M |
| Random Forest | **~7.65M (Best)** |

✔ Random Forest achieved the lowest error and most stable performance.

---

## 🏆 Final Model
- Algorithm: Random Forest Regressor
- Estimators: 100
- Evaluation Metric: RMSE
- Saved Artifacts:
  - `model.pkl`
  - `pipeline.pkl`

---

## 📦 Training & Inference Workflow

### Training Phase
1. Load and clean dataset
2. Perform stratified train-test split
3. Remove outliers (training data only)
4. Train model using preprocessing pipeline
5. Save trained model and pipeline

### Inference Phase
1. Load saved model and pipeline
2. Preprocess unseen test data
3. Generate price predictions
4. Save predictions to `output.csv`

---

## 🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

---


---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt

python main.py



