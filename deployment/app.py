import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Page Config ===
st.set_page_config(page_title="Weather Type Predictor", layout="centered")

# === Load Model and Encoders ===
@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f1, \
         open("le.pkl", "rb") as f2, \
         open("ohe_cat.pkl", "rb") as f3, \
         open("cloud_ohe.pkl", "rb") as f4, \
         open("scaler.pkl", "rb") as f5:
        model = pickle.load(f1)
        label_encoder = pickle.load(f2)
        ohe_cat = pickle.load(f3)  # for Season, Location
        cloud_ohe = pickle.load(f4)  # for Cloud Cover
        scaler = pickle.load(f5)
    return model, label_encoder, ohe_cat, cloud_ohe, scaler

model, le, ohe_cat, cloud_ohe, scaler = load_artifacts()

# === Input Fields ===
numerical_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
                  'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
categorical_cols = ['Season', 'Location']
cloud_col = 'Cloud Cover'

# === UI ===
st.title("🌤️ Weather Type Predictor")
st.markdown("Fill in the weather conditions below:")

with st.form("weather_form"):
    inputs = {}
    for col in numerical_cols:
        inputs[col] = st.number_input(col, format="%.2f", step=0.1)

    season = st.selectbox("🗓️ Season", ['Winter', 'Spring', 'Summer', 'Autumn'])
    location = st.selectbox("📍 Location", ['inland', 'mountain', 'coastal'])
    cloud_cover = st.selectbox("☁️ Cloud Cover", ['clear', 'cloudy', 'overcast', 'partly cloudy'])

    submit = st.form_submit_button("🌈 Predict Weather Type")

# === Prediction Logic ===
if submit:
    try:
        # Numerical Data
        num_data = np.array([[inputs[col] for col in numerical_cols]])

        # One-hot encode Season, Location
        cat_df = pd.DataFrame([[season, location]], columns=categorical_cols)
        cat_encoded = ohe_cat.transform(cat_df)
        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()

        # One-hot encode Cloud Cover
        cloud_df = pd.DataFrame([[cloud_cover]], columns=[cloud_col])
        cloud_encoded = cloud_ohe.transform(cloud_df)
        if hasattr(cloud_encoded, "toarray"):
            cloud_encoded = cloud_encoded.toarray()

        # Combine All Inputs
        full_input = np.hstack([num_data, cloud_encoded, cat_encoded])

        # Scale
        scaled_input = scaler.transform(full_input)

        # Predict
        prediction = model.predict(scaled_input)
        predicted_label = le.inverse_transform(prediction)[0]

        # Probabilities
        proba = model.predict_proba(scaled_input)[0]
        label_probs = {label: f"{100 * p:.2f}%" for label, p in zip(le.classes_, proba)}

        # Display Output
        st.success(f"🎯 Predicted Weather Type: **{predicted_label}**")

        st.subheader("📊 Prediction Probabilities")
        st.json(label_probs)

        st.subheader("📄 Input Summary")
        st.write("**Numerical Inputs:**")
        st.dataframe(pd.DataFrame(num_data, columns=numerical_cols))

        st.write("**Categorical Inputs:**")
        st.write(cat_df.assign(**{cloud_col: cloud_cover}))

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")


