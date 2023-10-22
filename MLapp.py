import streamlit as st
import joblib  # for loading your trained model
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import warnings
import os
import matplotlib
import plotly.figure_factory as ff


# Set a custom app title and icon
st.set_page_config(
    page_title="Brain Stroke Risk Prediction App",
    page_icon=":brain:",
    layout="wide",
)
# Define the Streamlit app layout
st.title("Brain Stroke Risk Prediction App")

# Load your trained machine learning model
model = joblib.load('best_gb_classifier_model.pkl')  # Update with your model's path

# Load data
if os.path.exists("C:\\Users\\Gihan\\Music\\ML Assignment"):
    df = pd.read_csv('sourcedatasetsaved.csv', index_col=None)


# Use st.selectbox to create tab selection
tabs = st.selectbox("Navigation", ["📈Stroke prediction", "🗃Data Profile Report", "ML Model Performance Evaluation"])

if tabs == "🗃Data Profile Report":
    st.title("Automated Exploratory Data Analysis With Pandas Profiling")
    profile_report = ProfileReport(df, explorative=True)
    st_profile_report(profile_report)
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

if tabs == "ML Model Performance Evaluation":
    # Define the confusion matrix data
    confusion_matrix_data = np.array([[816, 16], [36, 802]])

    # Define the Streamlit app layout
    st.title("Confusion Matrix")

    # Create a confusion matrix figure using Plotly
    fig = ff.create_annotated_heatmap(z=confusion_matrix_data, x=['Predicted Negative', 'Predicted Positive'],
                                      y=['Actual Negative', 'Actual Positive'])

    # Customize the layout of the confusion matrix chart (optional)
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )

    # Display the confusion matrix chart in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)


    # Define the Streamlit app layout
    st.title("Feature Importance of the ML Model")

    # Display an image from a URL
    st.image("C:\\Users\\Gihan\\Music\\ML Assignment\\Feature impotance plot.png", caption="Image Caption", use_column_width=True)

if tabs == "📈Stroke prediction":

    # Create input elements
    user_age = st.sidebar.number_input("Enter age:", min_value=0, max_value=99, step=1)
    user_gender = st.sidebar.selectbox("Select gender:", ["Male", "Female"])
    user_hypertension = st.sidebar.selectbox("Hypertension:", ["No", "Yes"])  # Binary input for hypertension
    user_heart_disease = st.sidebar.selectbox("Heart Disease:", ["No", "Yes"])  # Binary input for heart disease
    user_work_type = st.sidebar.selectbox("Work type:", ["Private", "Self-employed", "Govt_job", "children"])
    user_avg_glucose_level = st.sidebar.number_input("Enter average glucose level:", min_value=0.0, step=0.1)
    user_bmi = st.sidebar.number_input("Enter BMI:", min_value=0.0, step=0.1)
    user_smoking_status = st.sidebar.selectbox("Smoking status:", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    user_residence_type = st.sidebar.selectbox("Residence type:", ["Urban", "Rural"])

    # Create a button for making predictions
    predict_button = st.sidebar.button("Predict")

    # Initialize transformers for one-hot encoding and standard scaling
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    standard_scaler = StandardScaler()

    # Define categories for one-hot encoding
    gender_categories = ["Male", "Female"]
    work_type_categories = ["Private", "Self-employed", "Govt_job", "children"]
    smoking_status_categories = ["formerly smoked", "never smoked", "smokes", "Unknown"]
    residence_type_categories = ["Urban", "Rural"]

    # Fit the one-hot encoder with the predefined categories
    one_hot_encoder.fit(np.array(gender_categories).reshape(-1, 1))
    one_hot_encoder.fit(np.array(work_type_categories).reshape(-1, 1))
    one_hot_encoder.fit(np.array(smoking_status_categories).reshape(-1, 1))
    one_hot_encoder.fit(np.array(residence_type_categories).reshape(-1, 1))

    # Define a function to preprocess data and make predictions
    def preprocess_data(user_age, user_gender, user_hypertension, user_heart_disease, user_work_type, user_avg_glucose_level, user_bmi, user_smoking_status, user_residence_type):
        # Encode the categorical features using one-hot encoding
        gender_encoded = one_hot_encoder.transform(np.array(user_gender).reshape(1, -1))
        work_type_encoded = one_hot_encoder.transform(np.array(user_work_type).reshape(1, -1))
        smoking_status_encoded = one_hot_encoder.transform(np.array(user_smoking_status).reshape(1, -1))
        residence_type_encoded = one_hot_encoder.transform(np.array(user_residence_type).reshape(1, -1))

        # Preprocess the numerical features
        age = np.array(user_age).reshape(1, -1)
        avg_glucose_level = np.array(user_avg_glucose_level).reshape(1, -1)
        bmi = np.array(user_bmi).reshape(1, -1)

        # Encode binary inputs (0 or 1)
        hypertension_encoded = np.array([0 if user_hypertension == "No" else 1]).reshape(1, -1)
        heart_disease_encoded = np.array([0 if user_heart_disease == "No" else 1]).reshape(1, -1)

        # Fit the StandardScaler with the user input data
        scaled_age = standard_scaler.fit_transform(age)
        scaled_avg_glucose_level = standard_scaler.fit_transform(avg_glucose_level)
        scaled_bmi = standard_scaler.fit_transform(bmi)

        # Concatenate all preprocessed features into one array
        preprocessed_input = np.concatenate([scaled_age, hypertension_encoded, heart_disease_encoded,
                                             work_type_encoded, scaled_avg_glucose_level, scaled_bmi,
                                             smoking_status_encoded, gender_encoded, residence_type_encoded], axis=1)

        return preprocessed_input

    # Check if the user has input data and the prediction button is clicked
    if predict_button and user_age:
        preprocessed_input = preprocess_data(user_age, user_gender, user_hypertension, user_heart_disease, user_work_type,
                                              user_avg_glucose_level, user_bmi, user_smoking_status, user_residence_type)
        prediction = model.predict(preprocessed_input)
        if prediction[0] == 1:
            st.write(" ## High risk of stroke. Please consult your medical doctor.")
        else:
            st.write(" ## Low risk of stroke.")  # Assuming your model returns a single prediction

    st.subheader(' Health Tips to Prevent Stroke:')
    st.markdown('''##### Exercise regularly: Aim for at least 30 minutes of moderate-intensity exercise. :running: ''')
    st.markdown('''##### Eat a healthy diet: Eat plenty of fruits, vegetables, and whole grains. Limit processed foods, saturated and trans fats, and sodium.  :green_salad: ''')
    st.markdown('''##### Maintain a healthy weight. :swimmer:''')
    st.markdown('''##### Quit smoking. :no_smoking:''')
    st.markdown('''##### Control your blood pressure. :drop_of_blood:''')
    st.markdown('''##### Get regular checkups. :male-doctor::stethoscope:''')


