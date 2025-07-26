import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        model = joblib.load("XGBoost_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'XGBoost_model.pkl' not found. Please ensure the model is trained and saved.")
        return None

def create_preprocessing_pipeline():
    """Create the same preprocessing pipeline as used in the notebook"""
    # Define numerical and categorical columns as per the notebook
    numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod', 'tenure_group']
    
    # Numerical pipeline: median imputation + standard scaling
    numerical_pipeline = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    
    # Categorical pipeline: most frequent imputation + one-hot encoding
    categorical_pipeline = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ]
    
    # Create column transformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor

def encode_binary_features(data):
    """Apply the same binary encoding as done in the notebook"""
    
    # Binary mappings as per the notebook
    binary_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {'No': 0, 'Yes': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No phone service': 0, 'No': 0, 'Yes': 1},
        'OnlineSecurity': {'No internet service': 0, 'No': 0, 'Yes': 1},
        'OnlineBackup': {'No internet service': 0, 'No': 0, 'Yes': 1},
        'DeviceProtection': {'No internet service': 0, 'No': 0, 'Yes': 1},
        'TechSupport': {'No internet service': 0, 'No': 0, 'Yes': 1},
        'StreamingTV': {'No internet service': 0, 'No': 0, 'Yes': 1},
        'StreamingMovies': {'No internet service': 0, 'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    # Apply mappings
    for column, mapping in binary_mappings.items():
        if column in data.columns:
            data[column] = data[column].map(mapping)
    
    return data

def create_tenure_group(tenure):
    """Create tenure group as done in the notebook"""
    if tenure <= 12:
        return '0-12'
    elif tenure <= 24:
        return '12-24'
    elif tenure <= 48:
        return '24-48'
    else:
        return '48+'

def main():
    st.title("📊 Customer Churn Prediction")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    st.sidebar.header("Customer Information")
    st.sidebar.markdown("Please fill in the customer details below:")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        st.subheader("📞 Service Information")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        st.subheader("🔒 Security & Support")
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    
    with col2:
        st.subheader("🎬 Streaming Services")
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        
        st.subheader("💳 Account Information")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        st.subheader("💰 Financial Information")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=0.01)
    
    # Predict button
    if st.button("🔮 Predict Churn", type="primary"):
        try:
            # Create tenure group
            tenure_group = create_tenure_group(tenure)
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [senior_citizen],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'tenure_group': [tenure_group]
            })
            
            # Apply binary encoding
            input_data_encoded = encode_binary_features(input_data.copy())
            
            # Make prediction
            prediction = model.predict(input_data_encoded)
            prediction_proba = model.predict_proba(input_data_encoded)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_status = "Will Churn" if prediction[0] == 1 else "Will Stay"
                color = "red" if prediction[0] == 1 else "green"
                st.markdown(f"### Prediction: <span style='color: {color}'>{churn_status}</span>", unsafe_allow_html=True)
            
            with col2:
                churn_probability = prediction_proba[0][1] * 100
                st.metric("Churn Probability", f"{churn_probability:.1f}%")
            
            with col3:
                stay_probability = prediction_proba[0][0] * 100
                st.metric("Stay Probability", f"{stay_probability:.1f}%")
            
            # Add probability bar chart
            st.subheader("📊 Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Will Stay', 'Will Churn'],
                'Probability': [stay_probability, churn_probability]
            })
            
            st.bar_chart(prob_df.set_index('Outcome'))
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            if churn_probability >= 70:
                st.error("🚨 High Risk: This customer has a high probability of churning. Immediate retention efforts recommended!")
            elif churn_probability >= 50:
                st.warning("⚠️ Medium Risk: This customer shows signs of potential churn. Consider proactive retention measures.")
            else:
                st.success("✅ Low Risk: This customer is likely to stay. Continue providing excellent service.")
            
            # Feature importance (if available)
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                st.subheader("📈 Model Insights")
                st.info("This prediction is based on a trained XGBoost model with 80% accuracy on test data.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check that all inputs are valid and the model file is properly loaded.")

if __name__ == "__main__":
    main()
