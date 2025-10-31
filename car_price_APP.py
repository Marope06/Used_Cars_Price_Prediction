import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

# Load model
filename = r'C:\Users\StreamlitAPPs\Car_price_prediction\best_model.pkl'
with open(filename, 'rb') as file:
    load_model = pickle.load(file)

# Load dataset for sidebar options
df = pd.read_csv('car_price_dataset.csv', index_col=0)

# App title and image
st.title('Ray Platinum Wheels Used Car Price Prediction APP')
image = Image.open(r'C:\Users\StreamlitAPPs\Car_price_prediction\cars.png')
st.image(image, use_container_width=True)

# Show dataset sample
st.write('Sample of the dataset')
st.write(df.head())

# Sidebar for user inputs
st.sidebar.header("Enter the car's key features")
brand = st.sidebar.selectbox('Brand', df['brand'].unique())
model = st.sidebar.selectbox('Model', df['model'].unique())
vehicle_age = st.sidebar.slider('Vehicle Age', 0, 30, 1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, step=500)
seller_type = st.sidebar.selectbox('Seller Type', df['seller_type'].unique())
fuel_type = st.sidebar.selectbox('Fuel Type', df['fuel_type'].unique())
transmission_type = st.sidebar.selectbox('Transmission Type', df['transmission_type'].unique())
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, max_value=40.0, step=0.1)
engine = st.sidebar.number_input("Engine Capacity (CC)", min_value=500, max_value=7000, step=50)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=20.0, max_value=700.0, step=5.0)
seats = st.sidebar.slider('Seats', 1, 7, 5)

# Predict button
if st.button('Predict'):
    # Create input dataframe
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [model],
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

    #st.write("Input Data:", input_data)

    # Define categorical columns
    cat_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit encoder on categorical columns of the original dataset
    encoder.fit(df[cat_cols])

    # Transform input data
    encoded_array = encoder.transform(input_data[cat_cols])

    # Convert encoded array into DataFrame with correct column names
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))

    # Combine encoded and numeric columns
    input_encoded = pd.concat([input_data.drop(columns=cat_cols), encoded_df], axis=1)

    # Align input columns with model’s expected columns
    model_columns = load_model.feature_names_in_

    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training
    input_encoded = input_encoded[model_columns]

    #st.write("Encoded Data Ready for Prediction:")
    st.write(input_encoded.head())

    # Make prediction
    prediction = load_model.predict(input_encoded)
    st.success(f"Your car is estimated to sell for: R {round(prediction[0], 2)}")