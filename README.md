# Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization

## Overview

The **Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization** is an interactive web-based application built using **Streamlit**. It provides end-to-end solutions for data preprocessing, visualization, and machine learning model building. This tool is designed for both beginners and advanced users, simplifying complex tasks in the machine learning pipeline.

## Features

### File Support
- Supports **CSV**, **Excel**, and **JSON** file formats.
- Automatic data type inference and correction for seamless integration.

### Data Preprocessing
- **Missing Value Handling**: Options include dropping rows or filling with mean, median, or mode.
- **Outlier Detection**: Detect outliers using IQR, Z-score, or Isolation Forest methods.
- **Categorical Encoding**: Choose between **Label Encoding** or **One-Hot Encoding**.
- **Feature Scaling**: Support for standard, min-max, and robust scalers.
- **Time-Series Processing**: Automatically extracts year, month, and day features.

### Data Visualization
- **Numeric Columns**: Histogram, KDE, box plots, and pair plots.
- **Categorical Columns**: Bar plots and count plots.
- **Correlation Heatmaps**: Visualize relationships between numeric features.

### Machine Learning
- Supports both **Classification** and **Regression** tasks.
- Choose from a variety of models:
  - Random Forest, SVM, XGBoost, Neural Networks, Logistic Regression, and more.
- **Hyperparameter Tuning**: Built-in grid search for optimization.
- **Cross-Validation**: Evaluate models with 5-fold cross-validation.
- **Class Imbalance Handling**: Leverage SMOTE for oversampling.

### Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- **Regression Metrics**: MSE, RMSE, MAE, and R² Score.
- **ROC-AUC**: Binary and multi-class ROC curves with AUC scores.

### Advanced Features
- **PCA**: Dimensionality reduction with 2D visualization.
- **Text Preprocessing**: Tokenization, lemmatization, and stopword removal.
- **NLP Transformers**: Support for text classification using pre-trained models (e.g., DistilBERT).

### Save and Predict
- Save trained models, scalers, and encoders for future use.
- Upload new datasets for prediction with trained pipelines.

## How to run it in your environment?

  ```bash
   git clone https://github.com/your-username/file_name.git
   pip install -r requirements.txt
   streamlit run file_name.py
