import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)


def yes_no_to_binary_transformer(x):
    return (x == 'Yes').astype(int)


@st.cache_resource
def load_model():
    return joblib.load('logistic_model_pipeline.pkl')


model = load_model()

# Header
st.title("Telco Customer Churn Prediction")
st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("👥 Customer Demographics")
    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        partner = st.selectbox("Partner", ['Yes', 'No'])

    with demo_col2:
        senior_citizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
        dependents = st.selectbox("Dependents", ['Yes', 'No'])

    st.divider()

    st.subheader("📞 Phone Services")
    phone_col1, phone_col2 = st.columns(2)

    with phone_col1:
        phone_service = st.selectbox("Phone Service", ['Yes', 'No'])

    with phone_col2:
        if phone_service == 'Yes':
            multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No'])
        else:
            multiple_lines = 'No'
            st.selectbox("Multiple Lines", ['No'], disabled=True,
                         help="Phone service required for multiple lines")

    st.divider()

    st.subheader("🌐 Internet Services")
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    if internet_service != 'No':
        int_col1, int_col2 = st.columns(2)

        with int_col1:
            online_security = st.selectbox("Online Security", ['Yes', 'No'])
            online_backup = st.selectbox("Online Backup", ['Yes', 'No'])
            device_protection = st.selectbox("Device Protection", ['Yes', 'No'])

        with int_col2:
            tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
            streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No'])
            streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No'])
    else:
        online_security = 'No'
        online_backup = 'No'
        device_protection = 'No'
        tech_support = 'No'
        streaming_tv = 'No'
        streaming_movies = 'No'

        st.info("💡 Internet add-on services require an internet connection")

    st.divider()

    st.subheader("💳 Contract & Billing Information")

    contract_col1, contract_col2 = st.columns(2)

    with contract_col1:
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])

    with contract_col2:
        payment_method = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

    fin_col1, fin_col2 = st.columns(2)

    with fin_col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.01)

    with fin_col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=0.01)

with col2:
    st.subheader("📊 Customer Summary")

    services_count = sum([
        phone_service == 'Yes',
        multiple_lines == 'Yes',
        internet_service != 'No',
        online_security == 'Yes',
        online_backup == 'Yes',
        device_protection == 'Yes',
        tech_support == 'Yes',
        streaming_tv == 'Yes',
        streaming_movies == 'Yes'
    ])

    st.metric("Active Services", services_count, help="Total number of active services")
    st.metric("Customer Tenure", f"{tenure} months")
    st.metric("Monthly Revenue", f"${monthly_charges:.2f}")

    if tenure > 0:
        avg_monthly = total_charges / tenure
        st.metric("Avg Monthly Spend", f"${avg_monthly:.2f}")

    st.divider()

    st.info("📋 **Model Information**\n\n"
            "• Built with Logistic Regression\n"
            "• Trained on Telco Customer Dataset\n"
            "• Provides probability scores")

st.divider()
st.subheader("🎯 Churn Prediction")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("Predict Customer Churn", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.divider()

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ **HIGH CHURN RISK**")
                st.markdown("**Prediction: Customer will likely CHURN**")
            else:
                st.success("✅ **LOW CHURN RISK**")
                st.markdown("**Prediction: Customer will likely STAY**")

        with result_col2:
            st.metric("Churn Probability", f"{prediction_proba:.1%}",
                      delta=f"{prediction_proba - 0.5:.1%}" if prediction_proba != 0.5 else None)

        st.markdown("**Churn Risk Level:**")
        st.progress(prediction_proba, text=f"Risk Score: {prediction_proba:.1%}")

st.divider()
st.markdown("**🚀 Powered by Machine Learning** | Built with Streamlit")
st.caption(
    "This prediction tool helps identify customers at risk of churning based on their profile and service usage patterns.")
