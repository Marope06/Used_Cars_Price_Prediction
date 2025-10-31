import streamlit as st
import gzip
import pickle
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with gzip.open('best_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# -----------------------------
# Load Dataset for Sidebar Options
# -----------------------------
df = pd.read_csv('car_price_dataset.csv', index_col=0)

# -----------------------------
# App Title and Image
# -----------------------------
st.title('🚗 Ray Platinum Wheels Used Car Price Prediction APP')

url = "https://raw.githubusercontent.com/Marope06/Used_Cars_Price_Prediction/main/cars.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
st.image(image, use_container_width=True)

# Show dataset sample
st.write("### Dataset Sample")
st.write(df.head())

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter the car's key features")
brand = st.sidebar.selectbox('Brand', df['brand'].unique())
model_name = st.sidebar.selectbox('Model', df['model'].unique())
vehicle_age = st.sidebar.slider('Vehicle Age', 0, 30, 1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, step=500)
seller_type = st.sidebar.selectbox('Seller Type', df['seller_type'].unique())
fuel_type = st.sidebar.selectbox('Fuel Type', df['fuel_type'].unique())
transmission_type = st.sidebar.selectbox('Transmission Type', df['transmission_type'].unique())
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, max_value=40.0, step=0.1)
engine = st.sidebar.number_input("Engine Capacity (CC)", min_value=500, max_value=7000, step=50)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=20.0, max_value=700.0, step=5.0)
seats = st.sidebar.slider('Seats', 1, 7, 5)

# -----------------------------
# Prediction Section
# -----------------------------
if st.button('Predict'):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [model_name],
        'vehicle_age': [vehicle_age],
        'km_driven': [km_driven],
        'seller_type': [seller_type],
        'fuel_type': [fuel_type],
        'transmission_type': [transmission_type],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    # Define categorical columns
    cat_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']

    # Encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df[cat_cols])
    encoded_array = encoder.transform(input_data[cat_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))

    # Combine encoded and numeric features
    input_encoded = pd.concat([input_data.drop(columns=cat_cols), encoded_df], axis=1)

    # -----------------------------
    # Align Columns with Model
    # -----------------------------
    if hasattr(model, 'feature_names_in_'):
        model_columns = model.feature_names_in_
    else:
        # fallback (you can replace this with your actual feature list)
        model_columns = input_encoded.columns.tolist()

    # Add any missing columns
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns
    input_encoded = input_encoded[model_columns]

    st.write("### Encoded Input Data")
    st.write(input_encoded.head())

    # -----------------------------
    # Predict
    # -----------------------------
    prediction = model.predict(input_encoded)
    st.success(f"💰 Estimated Selling Price: R {round(prediction[0], 2)}")
