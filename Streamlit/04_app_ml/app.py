import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import io

# Step 1: Greeting the user
st.title("Machine Learning Model Trainer")
st.write("Welcome! This application allows you to train ML models on uploaded or example datasets.")

# Step 2: Data Upload or Example Data
data_choice = st.sidebar.radio("Select Dataset Option", ("Upload Data", "Use Example Data"))

@st.cache_data
def load_example_data(name):
    return sns.load_dataset(name)

data = None
if data_choice == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "tsv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)  # CSV
        except:
            try:
                data = pd.read_excel(uploaded_file)  # Excel
            except:
                data = pd.read_table(uploaded_file)  # TSV
else:
    dataset_name = st.sidebar.selectbox("Choose Example Dataset", ["titanic", "tips", "iris"])
    data = load_example_data(dataset_name)

# Step 3: User specifies if the problem is Regression or Classification
problem_type = st.radio("Select Problem Type", ["Regression", "Classification"])

# Step 4: Data Overview & Column Selection
if data is not None:
    st.write("### Data Overview")
    st.write("#### Head of the data:")
    st.dataframe(data.head())
    st.write(f"#### Shape: {data.shape}")
    st.write("#### Description:")
    st.write(data.describe())
    st.write("#### Column Names:")
    st.write(list(data.columns))

    # Step 5: Feature and Target Selection
    features = st.multiselect("Select Feature Columns", options=data.columns)
    target = st.selectbox("Select Target Column", options=data.columns)

    if target and features and st.button("Run Analysis"):
        df = data.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        
        encoders = {}
        for col in categorical_cols:
            if col in features:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))

        if df[target].dtype == 'object':
            target_encoder = LabelEncoder()
            df[target] = target_encoder.fit_transform(df[target].astype(str))
        
        imputer = IterativeImputer()
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        X = df[features]
        y = df[target]
        
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Step 6: Train-test split
        test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Step 7: Model Selection
        model_options = {
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Support Vector Machine": SVR()
            },
            "Classification": {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC()
            }
        }

        selected_model_name = st.sidebar.selectbox("Select Model", list(model_options[problem_type].keys()))
        selected_model = model_options[problem_type][selected_model_name]

        @st.cache_resource
        def train_model(model, X_train, y_train):
            model.fit(X_train, y_train)
            return model

        trained_model = train_model(selected_model, X_train, y_train)
        y_pred = trained_model.predict(X_test)

        # Step 8: Evaluation
        st.write("### Model Evaluation")
        if problem_type == "Regression":
            st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
            st.write(f"MAE: {mean_absolute_error(y_test, y_pred)}")
            st.write(f"R² Score: {r2_score(y_test, y_pred)}")
        else:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

        # Step 9: Download Model
        if st.button("Download Model"):
            buffer = io.BytesIO()
            pickle.dump(trained_model, buffer)
            st.download_button("Download Trained Model", buffer.getvalue(), file_name="model.pkl")

        # Step 10: Make Predictions
        if st.checkbox("Make Predictions"):
            input_data = []
            for col in features:
                input_data.append(st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())))
            input_data = np.array(input_data).reshape(1, -1)
            input_data = scaler.transform(input_data)
            prediction = trained_model.predict(input_data)
            st.write(f"Prediction: {prediction}")


