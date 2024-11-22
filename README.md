# Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization

## Overview
The **Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization** is an interactive Streamlit app that provides a comprehensive solution for analyzing datasets, preprocessing data, visualizing data, building machine learning models, and making predictions. It supports a wide range of preprocessing steps, including handling missing values, encoding categorical variables, detecting outliers, and applying feature scaling. The app also allows users to explore and visualize the data, including numeric and categorical features. After data preprocessing, users can build various machine learning models (classification and regression) and evaluate their performance and make predictions on new data.

## Features
- **Data Upload**: Upload a CSV file for analysis and model building.
- **Data Exploration**: 
  - View basic information about the dataset (columns, data types, missing values).
  - Visualize distributions of numeric and categorical columns.
  - Generate correlation heatmaps and pair plots for numeric columns.
- **Data Preprocessing**: 
  - Handle missing values using multiple strategies (drop, fill with mean, median, mode, etc.).
  - Detect and remove outliers using IQR, Z-Score, or Isolation Forest.
  - Encode categorical variables with Label Encoding or One-Hot Encoding.
  - Apply Principal Component Analysis (PCA) for dimensionality reduction.
  - NLP Preprocessing:
     - Preprocess text data by:
     - Lowercasing.
     - Removing punctuation and special characters.
     - Tokenizing the text.
     - Removing stopwords.
     - Lemmatizing words to their base form.
- **Model Building**: 
  - Automatically infer whether the task is classification or regression.
  - Choose from a wide range of machine learning algorithms, including:
    - Random Forest, XGBoost, LightGBM, CatBoost, SVM, Decision Tree, KNN, Logistic Regression, Gradient Boosting, AdaBoost, Naive Bayes, Neural Networks, and more.
  - Hyperparameter tuning using GridSearchCV.
  - Cross-validation for model evaluation.
- **Evaluation**: 
  - Comprehensive evaluation for classification tasks: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
  - Regression evaluation: MSE, MAE, RMSE, R2 Score.
- **Prediction**:
  - Make predictions on new data by uploading a separate CSV file with the same features.
  - Process and transform the new data before using the trained model to generate predictions.
  - View the predictions directly in the app.
