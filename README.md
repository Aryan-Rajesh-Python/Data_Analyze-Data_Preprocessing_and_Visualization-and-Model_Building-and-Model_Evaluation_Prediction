# Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization

## Overview

Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization is a platform built using Python. It leverages popular libraries such as pandas, numpy, scikit-learn, Streamlit, and others to provide an interactive interface for data loading, preprocessing, visualization, model building, and evaluation.

## Features
- **Data Loading and Basic Information**: Load CSV files and display basic information about the dataset, including the first few rows, summary statistics, and column types.
- **Data Visualization**: Visualize numeric and categorical columns using histograms, KDE plots, box plots, correlation heatmaps, and pair plots.
- **Missing Values Handling**: Handle missing values using various methods such as dropping rows, filling with mean, median, mode, or category-specific imputation.
- **Outlier Detection**: Detect and remove outliers using methods like IQR, Z-Score, and Isolation Forest.
- **Categorical Encoding**: Encode categorical columns using Label Encoding or One-Hot Encoding.
- **PCA Analysis**: Perform Principal Component Analysis (PCA) for dimensionality reduction and visualize the first two principal components.
- **Model Building and Evaluation**: Build and evaluate various machine learning models (e.g., Random Forest, SVM, Decision Tree, XGBoost, KNN, Logistic Regression, Gradient Boosting, etc.) with options for hyperparameter tuning.
- **Text Preprocessing**: Preprocess text data for NLP tasks, including tokenization, stopword removal, and lemmatization.
- **Interactive Streamlit Interface**: Provide an interactive interface for users to upload datasets, visualize data, handle missing values, detect outliers, encode categorical columns, perform PCA, build and evaluate models, and make predictions on new data.
- **New Data Prediction**: Upload new data and generate predictions using the trained model, with support for preprocessing and feature scaling.

## Installation
To run this project, you need to have Python installed on your system. You can install the required libraries using the following command:
```bash
pip install pandas numpy streamlit scikit-learn xgboost lightgbm catboost seaborn matplotlib nltk transformers imblearn
```

## Usage
1. **Run the Streamlit App**:
   ```bash
   streamlit run Data_Analyzer.py
   ```

2. **Upload Dataset**:
   - Upload your dataset in CSV format using the file uploader.

3. **Data Analysis and Preprocessing**:
   - View basic information about the dataset.
   - Visualize numeric and categorical columns.
   - Handle missing values and detect outliers.
   - Encode categorical columns and perform PCA analysis.

4. **Model Building**:
   - Select the target column and choose a machine learning model.
   - Optionally, tune hyperparameters using GridSearchCV.
   - Evaluate the model using cross-validation and various metrics.

5. **Make Predictions**:
   - Upload new data for prediction and view the predicted results.

## New Data Prediction
The Data Analyzer platform allows you to make predictions on new data using the trained model. Follow these steps to generate predictions:

1. **Upload New Data**:
   - Use the file uploader to upload a new dataset in CSV format.

2. **View New Data**:
   - The uploaded new data will be displayed for review.

3. **Generate Predictions**:
   - The platform will preprocess the new data, apply any necessary feature scaling, and generate predictions using the trained model.
   - The predictions will be displayed in the interface.

## Functions
- `load_data(uploaded_file)`: Loads the dataset from a CSV file.
- `basic_info(df)`: Displays basic information about the dataset.
- `visualize_columns(df, max_categories=10, figsize=(12, 10), max_pairplot_cols=10)`: Visualizes numeric and categorical columns.
- `handle_missing_values(df)`: Handles missing values using various methods.
- `detect_outliers(df)`: Detects and removes outliers using different methods.
- `encode_categorical(df)`: Encodes categorical columns using Label Encoding or One-Hot Encoding.
- `pca_analysis(df)`: Performs PCA analysis and visualizes the first two principal components.
- `evaluate_with_cross_validation(model, X, y, task_type="classification")`: Evaluates model performance using cross-validation.
- `preprocess_text(text)`: Preprocesses text data for NLP tasks.
- `build_ml_model(df, target_column)`: Builds and evaluates a machine learning model.
- `predict_new_data(model, input_data, label_encoders=None, scaler=None)`: Generates predictions on new data.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- pandas
- numpy
- scikit-learn
- Streamlit
- xgboost
- lightgbm
- catboost
- seaborn
- matplotlib
- nltk
- transformers
- imblearn
- io
