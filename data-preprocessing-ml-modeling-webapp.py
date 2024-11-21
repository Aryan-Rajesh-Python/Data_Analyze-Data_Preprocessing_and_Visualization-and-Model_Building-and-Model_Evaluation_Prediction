import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
import lightgbm as lgb
import catboost as cb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, StackingClassifier
from transformers import pipeline

# Load the dataset
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid CSV file.")
            return None
        st.write(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or not in CSV format. Please upload a valid CSV.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Ensure it is properly formatted.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return None

# Basic info about the dataset
def basic_info(df):
    st.subheader("Basic Information")
    st.write("### First 5 rows of the dataset:")
    st.dataframe(df.head())  # Display first 5 rows
    st.write("### Dataset Overview:")
    st.table(df.describe())  # Summary statistics for numeric columns
    st.write("### Column Types and Missing Values:")
    st.table(df.info())  # Data types and missing values

# Visualize columns
def visualize_columns(df, max_categories=10, figsize=(12, 10)):
    st.subheader("Column Visualizations")
    
    # Numeric Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    st.write("### Distribution Plots (Histogram and KDE) for Numeric Columns:")
    for col in numeric_cols:
        fig, ax = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
        
        # Histogram with larger size
        ax[0].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        ax[0].set_title(f"Histogram of {col}")
        ax[0].set_xlabel(col)
        ax[0].set_ylabel('Frequency')
        
        # KDE Plot
        sns.kdeplot(df[col], ax=ax[1], color='red')
        ax[1].set_title(f"KDE of {col}")
        ax[1].set_xlabel(col)
        ax[1].set_ylabel('Density')
        
        st.pyplot(fig)

    # Box Plots
    st.write("### Box Plots for Numeric Columns:")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap for Numeric Columns:")
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(figsize[0]*1.5, figsize[1]*1.5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Pair Plot (only for numeric columns)
    st.write("### Pair Plot (For Numeric Columns):")
    pair_plot = sns.pairplot(df[numeric_cols])
    pair_plot.fig.set_size_inches(figsize[0]*1.5, figsize[1]*1.5)
    st.pyplot(pair_plot.fig)

    # Categorical Columns Visualizations
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        st.write(f"### Visualizations for Categorical Column: {col}")
        
        # Limit categories to top N most frequent
        top_categories = df[col].value_counts().nlargest(max_categories).index
        df_filtered = df[df[col].isin(top_categories)]
        
        # Bar Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=df_filtered[col].value_counts().index, 
                    y=df_filtered[col].value_counts().values, ax=ax)
        ax.set_title(f"Barplot of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Count Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x=df_filtered[col], ax=ax)
        ax.set_title(f"Countplot of {col}")
        st.pyplot(fig)

# Handle missing values
def handle_missing_values(df):
    st.subheader("Handle Missing Values")
    st.write("### Choose the method to handle missing values:")
    missing_method = st.selectbox("Select Method", ["Drop", "Fill Mean", "Fill Median", "Impute"])
    
    if missing_method == "Drop":
        df_cleaned = df.dropna()
    elif missing_method == "Fill Mean":
        df_cleaned = df.fillna(df.mean())
    elif missing_method == "Fill Median":
        df_cleaned = df.fillna(df.median())
    elif missing_method == "Impute":
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    st.write(f"### Missing values handled using {missing_method}.")
    return df_cleaned

# Detect outliers using IQR
def detect_outliers(df):
    st.subheader("Outlier Detection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_method = st.selectbox("Select Outlier Detection Method", ["IQR", "None"])

    df_outliers_removed = df.copy()
    if outlier_method == "IQR":
        Q1 = df_outliers_removed[numeric_cols].quantile(0.25)
        Q3 = df_outliers_removed[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_outliers_removed[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                    (df_outliers_removed[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        df_outliers_removed = df_outliers_removed[~outliers]
        st.write(f"Outliers removed using the IQR method. {outliers.sum()} rows removed.")
    else:
        st.write("No outlier removal applied.")

    return df_outliers_removed

# Encode categorical columns
def encode_categorical(df):
    st.subheader("Encode Categorical Columns")
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        label_encoders[col] = encoder
    st.write("### Categorical columns encoded successfully.")
    return df, label_encoders

# PCA analysis
def pca_analysis(df):
    st.subheader("Principal Component Analysis (PCA)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(df[numeric_cols])
        df_pca = pd.DataFrame(pca_components, columns=["PCA 1", "PCA 2"])
        st.write("### 2D PCA visualization:")
        st.dataframe(df_pca.head())
        
        fig, ax = plt.subplots()
        ax.scatter(df_pca['PCA 1'], df_pca['PCA 2'])
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        st.pyplot(fig)
    else:
        st.write("### PCA is not applicable. More than one numerical column is required.")

def evaluate_with_cross_validation(model, X, y, task_type="classification"):
    """
    Perform cross-validation to evaluate model performance
    """
    if task_type == "classification":
        scoring = 'accuracy'  # For classification tasks, use accuracy
    else:
        scoring = 'neg_mean_squared_error'  # For regression tasks, use negative MSE

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    
    # Output the results
    st.write(f"Cross-validation scores: {cv_scores}")
    st.write(f"Mean CV score: {cv_scores.mean():.2f}")
    st.write(f"Standard deviation of CV scores: {cv_scores.std():.2f}")
    
def build_ml_model(df, target_column):
    # Split the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    if X.isnull().sum().any() or y.isnull().sum() > 0:
        st.warning("The dataset contains missing values. Consider handling them before proceeding.")
        return None, None
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling for models that require it
    scaler = None
    if st.selectbox('Do you want to scale features?', ['No', 'Yes']) == 'Yes':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Select task type: classification or regression
    task_type = st.selectbox('Select Task Type:', ['classification', 'regression'])

    # Choose the model
    model_type = st.selectbox("Choose the model", [
    "Random Forest", "SVM", "Decision Tree", "XGBoost", "KNN", "Logistic Regression", 
    "Gradient Boosting", "Linear Regression", "Naive Bayes", "AdaBoost", "CatBoost", 
    "LightGBM", "Ridge Regression", "Lasso Regression", "ElasticNet", "Neural Network",
    "Voting Classifier", "Stacking Classifier", "NLP Transformer"
    ])

    model = None
    if model_type == "Random Forest":
        model = RandomForestClassifier() if task_type == "classification" else RandomForestRegressor()
    elif model_type == "SVM":
        model = SVC() if task_type == "classification" else SVR()
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier() if task_type == "classification" else DecisionTreeRegressor()
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier() if task_type == "classification" else xgb.XGBRegressor()
    elif model_type == "KNN":
        model = KNeighborsClassifier() if task_type == "classification" else KNeighborsRegressor()
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier() if task_type == "classification" else GradientBoostingRegressor()
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Naive Bayes":
        model = GaussianNB() if task_type == "classification" else None
    elif model_type == "AdaBoost":
        model = AdaBoostClassifier() if task_type == "classification" else AdaBoostRegressor()
    elif model_type == "CatBoost":
        model = cb.CatBoostClassifier() if task_type == "classification" else cb.CatBoostRegressor()
    elif model_type == "LightGBM":
        model = lgb.LGBMClassifier() if task_type == "classification" else lgb.LGBMRegressor()
    elif model_type == "Ridge Regression":
        model = Ridge()
    elif model_type == "Lasso Regression":
        model = Lasso()
    elif model_type == "ElasticNet":
        model = ElasticNet()
    elif model_type == "Neural Network":
        model = MLPClassifier() if task_type == "classification" else MLPRegressor()
    elif model_type == "Voting Classifier":
        model = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier()), 
            ('svc', SVC(probability=True)), 
            ('gb', GradientBoostingClassifier())
        ], voting='soft')
    elif model_type == "Stacking Classifier":
        model = StackingClassifier(estimators=[
            ('rf', RandomForestClassifier()), 
            ('svc', SVC(probability=True)), 
            ('gb', GradientBoostingClassifier())
        ], final_estimator=LogisticRegression())
    elif model_type == "NLP Transformer":
        st.warning("Using a pre-trained transformer model for text classification.")
        model = pipeline('text-classification', model='distilbert-base-uncased')
        
    if model is None:
        st.error("Invalid model type selected.")
        return None, None

    # Hyperparameter tuning
    param_grid = {}
    if st.checkbox("Do you want to tune hyperparameters?"):
        # Customize the hyperparameter grid based on the selected model
        if model_type == "Random Forest":
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]}
        elif model_type == "SVM":
            param_grid = {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf']}
        elif model_type == "Decision Tree":
            param_grid = {"max_depth": [5, 10, 15], "min_samples_split": [2, 5, 10]}
        elif model_type == "XGBoost":
            param_grid = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2]}
        elif model_type == "KNN":
            param_grid = {"n_neighbors": [3, 5, 7], "weights": ['uniform', 'distance']}
        elif model_type == "Logistic Regression":
            param_grid = {"C": [0.1, 1, 10], "solver": ['liblinear', 'saga']}
        elif model_type == "Gradient Boosting":
            param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        elif model_type == "Linear Regression":
            pass
        elif model_type == "Naive Bayes":
            pass
        elif model_type == "AdaBoost":
            param_grid = {"n_estimators": [50, 100, 200]}
        elif model_type == "CatBoost":
            param_grid = {"iterations": [100, 200], "learning_rate": [0.01, 0.1]}
        elif model_type == "LightGBM":
            param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        elif model_type == "Ridge Regression":
            param_grid = {"alpha": [0.1, 1, 10]}
        elif model_type == "Lasso Regression":
            param_grid = {"alpha": [0.1, 1, 10]}
        elif model_type == "ElasticNet":
            param_grid = {"alpha": [0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]}
        elif model_type == "Neural Network":
            param_grid = {"hidden_layer_sizes": [(50,), (100,)], "activation": ['relu', 'tanh']}
        elif model_type == "Voting Classifier":
            param_grid = {
                "weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1]],  # Weight combinations for rf, gb, svc
                "voting": ["soft", "hard"]
            }
        elif model_type == "Stacking Classifier":
            param_grid = {
                "final_estimator": [LogisticRegression(), RandomForestClassifier()],
                "cv": [3, 5]  # Cross-validation splitting strategy
            }
        elif model_type == "NLP Transformer":
            param_grid = {
                "learning_rate": [1e-5, 3e-5, 5e-5],
                "num_train_epochs": [2, 3, 5],
                "batch_size": [16, 32]
            }

    grid_search = GridSearchCV(model, param_grid, cv=5)

    try:
        # Fit the model and find the best parameters
        grid_search.fit(X_train, y_train)
        st.write(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    except Exception as e:
        st.error(f"An error occurred while fitting the model: {e}")
        return
    
    # Display the best model
    st.write(f"### Best Model: {grid_search.best_estimator_}")
    
    # Evaluate with Cross-validation
    evaluate_with_cross_validation(grid_search.best_estimator_, X_train, y_train, task_type)

    # Make predictions
    y_pred = grid_search.predict(X_test)

    # Evaluate model performance
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"### Accuracy: {accuracy:.2f}")
        
        # Classification Report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        
        st.write("### Classification Report:")
        st.dataframe(class_report_df)

        # Confusion Matrix
        st.write("### Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        st.pyplot(fig)

    else:  # Regression tasks
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"### Mean Squared Error: {mse:.2f}")
        st.write(f"### R2 Score: {r2:.2f}")

        # Cross-validation score for better performance estimate
        cv_score = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='neg_mean_squared_error')
        st.write(f"### Cross-Validation Score: {-np.mean(cv_score):.2f} (Negative MSE)")

        # Actual vs Predicted plot
        st.write("### Actual vs Predicted Plot:")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)
        
    return model, scaler

# Prediction Function
def predict_new_data(model, input_data, label_encoders=None, scaler=None):
    # Apply label encoding to categorical columns
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col].astype(str))

    # Apply feature scaling if a scaler was used
    if scaler:
        input_data = scaler.transform(input_data)

    # Generate predictions
    predictions = model.predict(input_data)
    return predictions

def main():
    st.title("Machine Learning Model Builder and Analyzer")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            basic_info(df)
            visualize_columns(df)
            df_cleaned = handle_missing_values(df)
            df_outliers_removed = detect_outliers(df_cleaned)
            df_encoded, label_encoders = encode_categorical(df_outliers_removed)
            pca_analysis(df_encoded)
            target_column = st.selectbox("Select the target column", df_encoded.columns)
            model, scaler = build_ml_model(df_encoded, target_column)
            
            # Allow user to upload new data for prediction
            st.subheader("Make Predictions on New Data")
            new_data_file = st.file_uploader("Upload new data for prediction (CSV)", type=["csv"])
            if new_data_file is not None:
                new_data = pd.read_csv(new_data_file)
                st.write("### New Data:")
                st.dataframe(new_data)
                
                try:
                    # Make predictions
                    predictions = predict_new_data(model, new_data, label_encoders, scaler)
                    st.write("### Predictions:")
                    st.write(predictions)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
