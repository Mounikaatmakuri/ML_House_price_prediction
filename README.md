## 🏡 House Rent Prediction Using Machine Learning and Deep Learning
# 📌 Overview
This project builds a predictive modeling pipeline to estimate house rent prices using various machine learning and deep learning algorithms, including Linear Regression, Ridge, Lasso, SVR, Random Forest, XGBoost, CatBoost, and an Artificial Neural Network (ANN). The goal is to provide accurate rent predictions based on property and locality features.

# 📊 Problem Statement
House rent prices vary significantly across locations and property characteristics. An intelligent rent prediction system helps:

Renters find fair deals

Property owners price competitively

Platforms provide better search and recommendations

# 📁 Dataset
Source: House_Rent_Dataset.csv

Shape: 4746 rows × 12 columns

Key Features:

BHK, Size, Floor, Area Type, City, Furnishing Status, Tenant Preferred, Bathroom, etc.

Target: Rent

# 🔧 Tools & Technologies
Languages & Libraries: Python, NumPy, Pandas, Matplotlib, Seaborn, Plotly, SciPy

ML Models: Ridge, Lasso, Random Forest, Gradient Boosting, SVR, KNN, XGBoost, CatBoost, Bayesian Ridge

Deep Learning: TensorFlow / Keras

Preprocessing: StandardScaler, Box-Cox Transformation

Validation: K-Fold Cross Validation

# 🧪 Data Processing Pipeline
Data Cleaning

Handled duplicates

Dropped outliers (extreme rent values)

Converted Floor into Floor Level and Total Floors

Extracted date features from Posted On

Encoding

Applied one-hot encoding to categorical variables

Removed high-cardinality feature Area Locality

Feature Scaling

Standardized numeric features using StandardScaler

Applied Box-Cox transformation to the Rent variable for normality

# 🏗️ Modeling
Traditional Models:
Trained 9 regressors:

Ridge, Lasso, Bayesian Ridge

KNN, SVR, Random Forest

Gradient Boosting, XGBoost, CatBoost

Deep Learning Model:
Created a fully connected ANN with:

Layers: 20 → 60 → 60 → 60 → 1

Loss: Mean Squared Error

Optimizer: Adam

Metric: RMSE

Trained for 5 epochs with validation

# 📉 Evaluation Metrics
Cross-Validation Metric: RMSE (Root Mean Squared Error)

ANN RMSE on Test Set: ~0.0191

Ensemble Model RMSE: ~50,367

R² Score (Ensemble): 0.55

📊 Results
Model	RMSE
Support Vector	0.0444
Lasso	0.0301
ANN	0.0191
KNN	0.0160
Ridge	0.0144
Bayesian Ridge	0.0144
XGBoost	0.0141
Random Forest	0.0141
Gradient Boosting	0.0136
CatBoost	0.0133 (lowest)

# 🎯 Ensemble Learning
A final prediction was made by averaging four top models:

XGBoost

CatBoost

Gradient Boosting

Random Forest

# 📈 Final Performance:
RMSE: 50,368

R² Score: 0.55

