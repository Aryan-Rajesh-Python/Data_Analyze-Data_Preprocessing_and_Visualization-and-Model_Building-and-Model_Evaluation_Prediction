import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report, mean_absolute_error
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import io
import pickle

# Load the dataset
def load_data(uploaded_file):
    """
    Load dataset and handle multiple file formats (CSV, Excel, JSON).
    Infer and correct data types automatically.
    """
    try:
        # Load based on file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON files.")
            return None
        
        # Check for empty file
        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid file.")
            return None

        # Infer and correct data types
        df = infer_and_correct_data_types(df)
        
        st.write(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def infer_and_correct_data_types(df):
    """
    Attempt to infer and correct data types in the dataframe.
    Example: Convert date columns from object to datetime, etc.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try converting to datetime
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                pass  # If not a datetime column, skip
        elif df[col].dtype == 'int64' or df[col].dtype == 'float64':
            # Handle mixed-type numeric columns (e.g., containing NaNs as objects)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Basic info about the dataset
def basic_info(df):
    st.subheader("Basic Information")
    st.write("### First 5 rows of the dataset:")
    st.dataframe(df.head())  # Display first 5 rows
    st.write("### Dataset Overview:")
    st.table(df.describe())  # Summary statistics for numeric columns
    
    # Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.write("### Column Types and Missing Values:")
    st.text(info)
    
# Visualize columns
def visualize_columns(df, max_categories=10, figsize=(12, 10), max_pairplot_cols=10):
    st.subheader("Column Visualizations")
    
    # Numeric Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    st.write("### Distribution Plots (Histogram and KDE) for Numeric Columns:")
    for col in numeric_cols:
        fig, ax = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
        
        # Histogram
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
        ax.set_xlabel(col)
        st.pyplot(fig)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap for Numeric Columns:")
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(figsize[0]*1.5, figsize[1]*1.5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # Pair Plot
    st.write("### Pair Plot (For Numeric Columns):")
    if len(numeric_cols) > max_pairplot_cols:
        st.warning(f"Too many numeric columns ({len(numeric_cols)}). Showing pair plot for the first {max_pairplot_cols} columns.")
        numeric_cols = numeric_cols[:max_pairplot_cols]
    if len(numeric_cols) > 1:
        pair_plot = sns.pairplot(df[numeric_cols])
        pair_plot.fig.suptitle("Pair Plot", y=1.02)  # Add a title
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
        sns.countplot(x=df_filtered[col], ax=ax, order=top_categories)
        ax.set_title(f"Countplot of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Handle missing values
def handle_missing_values(df):
    st.subheader("Handle Missing Values")
    st.write("### Choose the method to handle missing values:")
    missing_method = st.selectbox("Select Method", ["Drop", "Fill Mean", "Fill Median", "Fill Mode", "Category-Specific Imputation"])
    
    # Initialize df_cleaned
    df_cleaned = df.copy()

    if missing_method == "Drop":
        df_cleaned = df_cleaned.dropna()
    elif missing_method == "Fill Mean":
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    elif missing_method == "Fill Median":
        df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
    elif missing_method == "Fill Mode":
        for col in df_cleaned.columns:
            mode = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode, inplace=True)
    elif missing_method == "Category-Specific Imputation":
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':  # Categorical columns
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            else:  # Numeric columns
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    st.write(f"### Missing values handled using {missing_method}.")
    return df_cleaned

def handle_high_cardinality(df, max_categories=10):
    """
    Bucket rare categories for high-cardinality categorical columns.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() > max_categories:
            top_categories = df[col].value_counts().nlargest(max_categories).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    return df

# Detect outliers using IQR
def detect_outliers(df):
    
    st.subheader("Outlier Detection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_method = st.selectbox("Select Outlier Detection Method", ["IQR", "Z-Score", "Isolation Forest", "None"])
    df_outliers_removed = df.copy()
    
    if outlier_method == "IQR":
        Q1 = df_outliers_removed[numeric_cols].quantile(0.25)
        Q3 = df_outliers_removed[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_outliers_removed[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                    (df_outliers_removed[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        df_outliers_removed = df_outliers_removed[~outliers]
        st.write(f"Outliers removed using the IQR method. {outliers.sum()} rows removed.")
    
    elif outlier_method == "Z-Score":
        z_scores = np.abs((df_outliers_removed[numeric_cols] - df_outliers_removed[numeric_cols].mean()) /
                          df_outliers_removed[numeric_cols].std())
        outliers = (z_scores > 3).any(axis=1)
        df_outliers_removed = df_outliers_removed[~outliers]
        st.write(f"Outliers removed using the Z-Score method. {outliers.sum()} rows removed.")
    
    elif outlier_method == "Isolation Forest":
        iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        predictions = iso.fit_predict(df_outliers_removed[numeric_cols])
        df_outliers_removed = df_outliers_removed[predictions == 1]
        st.write("Outliers removed using Isolation Forest with parallel processing.")

    else:
        st.write("No outlier removal applied.")
    
    return df_outliers_removed

# Encode categorical columns
def encode_categorical(df):
    st.subheader("Encode Categorical Columns")
    encoding_method = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
    label_encoders = {}
    
    if encoding_method == "Label Encoding":
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            label_encoders[col] = encoder
        st.write("### Categorical columns encoded using Label Encoding.")
    
    elif encoding_method == "One-Hot Encoding":
        df = pd.get_dummies(df, drop_first=True)
        st.write("### Categorical columns encoded using One-Hot Encoding.")
    
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
        
def handle_time_series(df):
    """
    Detect and preprocess time-series data.
    """
    date_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
    if date_cols:
        st.write("### Detected Time-Series Columns:")
        st.write(date_cols)
        for col in date_cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df.drop(columns=[col], inplace=True)
        st.write("### Time-series columns transformed into year, month, day features.")
    return df

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
    
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Rejoin words into a single string
    return ' '.join(words)
    
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
    
    # Initialize scaler to None by default
    scaler = None

    # Feature scaling for models that require it
    if st.selectbox('Do you want to scale features?', ['No', 'Yes']) == 'Yes':
        scaler_option = st.selectbox("Choose a scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        if scaler_option == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif scaler_option == "RobustScaler":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Automatically infer task type
    if y.dtype == 'object' or len(y.unique()) <= 10:
        task_type = 'classification'
    else:
        task_type = 'regression'

    st.write(f"Task type inferred as: {task_type}")

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
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=10)), 
            ('svc', SVC(probability=True)), 
            ('gb', GradientBoostingClassifier())
        ]
        model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())
        st.write("Default Stacking Classifier configured with Logistic Regression as the meta-model.")
    elif model_type == "NLP Transformer":
        # Select the text column
        text_columns = X.select_dtypes(include=['object']).columns
        if len(text_columns) == 1:
            text_column = text_columns[0]
        elif len(text_columns) > 1:
            text_column = st.selectbox("Select the text column for NLP tasks", text_columns)
        else:
            st.error("No text column found for NLP Transformer. Please ensure your dataset contains at least one text column.")
            return None, None

        # Apply preprocessing
        st.write(f"Applying text preprocessing to the column: {text_column}")
        X[text_column] = X[text_column].apply(preprocess_text)
        
        # Initialize the NLP model
        try:
            model = pipeline('text-classification', model='distilbert-base-uncased')
            st.write("NLP Transformer initialized for text classification.")
        except Exception as e:
            st.error(f"Error loading NLP transformer: {e}. Using default configuration.")
            return None, None

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
                "learning_rate": [1e-6, 1e-5, 3e-5, 5e-5, 1e-4],
                "num_train_epochs": [2, 3, 4, 5, 6],
                "batch_size": [8, 16, 32, 64],
                "warmup_steps": [0, 500, 1000],
                "weight_decay": [0.01, 0.1],
                "adam_epsilon": [1e-8, 1e-5]
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
    
    if task_type == 'classification':        # Classification tasks
        # Handle class imbalance using SMOTE
        if st.checkbox("Handle imbalanced data?"):
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.write(f"SMOTE applied: New training set size - {X_train.shape[0]} samples.")

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
        recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
        f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        # Display metrics
        st.write(f"### Accuracy: {accuracy:.2f}")
        st.write(f"### Precision: {precision:.2f}")
        st.write(f"### Recall: {recall:.2f}")
        st.write(f"### F1 Score: {f1:.2f}")
        st.write(f"### Macro-Average Precision: {precision_macro:.2f}")
        st.write(f"### Macro-Average Recall: {recall_macro:.2f}")

        # Classification Report
        st.write("### Classification Report:")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(class_report_df)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        st.pyplot(fig)

        # ROC-AUC for classification tasks
        if len(np.unique(y_test)) == 2:  # Binary classification
            from sklearn.metrics import roc_auc_score, roc_curve
            y_prob = grid_search.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            st.write(f"### ROC-AUC: {roc_auc:.2f}")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_title('Receiver Operating Characteristic Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)

        else:  # Multi-class classification
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            y_prob = grid_search.predict_proba(X_test)
            n_classes = y_test_bin.shape[1]

            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fig, ax = plt.subplots(figsize=(10, 8))
            for i in range(n_classes):
                ax.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_title('Multi-Class Receiver Operating Characteristic Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)

    else:  # Regression tasks
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display regression metrics
        st.write(f"### Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"### Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"### Root Mean Squared Error (RMSE): {rmse:.2f}")
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
    """
    Generate predictions for new data using a trained model, with preprocessing.
    """
    try:
        # Ensure input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # Handle missing values in new data (default to mode or mean)
        for col in input_data.columns:
            if input_data[col].isnull().sum() > 0:
                if input_data[col].dtype == 'object':
                    input_data[col].fillna(input_data[col].mode()[0], inplace=True)
                else:
                    input_data[col].fillna(input_data[col].mean(), inplace=True)

        # Apply label encoding to categorical columns
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in input_data.columns:
                    input_data[col] = encoder.transform(input_data[col].astype(str))

        # Apply feature scaling if a scaler was used during training
        if scaler:
            input_data = scaler.transform(input_data)

        # Generate predictions
        predictions = model.predict(input_data)

        return predictions

    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

def save_pipeline(model, scaler, label_encoders):
    """
    Save the model, scaler, and encoders for future predictions.
    """
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    st.success("Pipeline saved for future use.")

def load_pipeline():
    """
    Load the saved model, scaler, and encoders.
    """
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None, None, None

def main():
    st.title("Comprehensive Data Analysis, Preprocessing, Machine Learning Model Building, and Prediction Platform with Interactive Visualization")
    
    # Step 1: File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])
    if uploaded_file is not None:
        # Step 2: Load data with type correction
        df = load_data(uploaded_file)
        if df is not None:
            # Step 3: Basic info and visualization
            basic_info(df)
            visualize_columns(df)

            # Step 4: Preprocessing pipeline
            df = infer_and_correct_data_types(df)  # Infer and correct types
            df_cleaned = handle_missing_values(df)  # Handle missing values
            df_outliers_removed = detect_outliers(df_cleaned)  # Handle outliers
            df_outliers_removed = handle_high_cardinality(df_outliers_removed)  # Handle high cardinality
            df_encoded, label_encoders = encode_categorical(df_outliers_removed)  # Encode categorical variables
            df_encoded = handle_time_series(df_encoded)  # Process time-series columns
            
            # Step 5: PCA Analysis
            pca_analysis(df_encoded)

            # Step 6: Target column selection
            target_candidates = [
                col for col in df_encoded.columns
                if df_encoded[col].nunique() < 50 or df_encoded[col].dtype in ['int', 'float']
            ]
            target_column = st.selectbox("Select the target column", target_candidates)
            if not target_column:
                st.warning("No valid target column found. Please ensure your dataset contains suitable columns.")
                return  # Exit if no target column is found

            # Step 7: Build the model
            model, scaler = build_ml_model(df_encoded, target_column)
            if model:
                if st.checkbox("Save the model and pipeline?"):
                    save_pipeline(model, scaler, label_encoders)

            # Step 8: Allow predictions on new data
            st.subheader("Make Predictions on New Data")
            new_data_file = st.file_uploader("Upload new data for prediction (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])
            if new_data_file is not None:
                try:
                    # Load new data
                    new_data = load_data(new_data_file)
                    st.write("### New Data:")
                    st.dataframe(new_data)

                    # Make predictions
                    predictions = predict_new_data(model, new_data, label_encoders, scaler)
                    st.write("### Predictions:")
                    st.write(predictions)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
