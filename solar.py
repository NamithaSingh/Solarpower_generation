# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:51:31 2024

@author: Namitha Singh
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('solarpower_trained_model.pkl', 'rb'))

# Function for prediction
def solarpower_generation_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Cap the prediction at 0
    prediction_capped = max(prediction[0], 0)
    
    return prediction_capped

def main():
    # Title of the app
    st.title('Solar Power Generation Prediction Web App')
    
    # Getting the input data from the user
    distance_to_solar_noon = st.text_input('Distance-to-Solar-noon Value (in radians)')
    temperature = st.text_input('Temperature (in degrees Celsius)')
    wind_direction = st.text_input('Wind Direction (in degrees)')
    wind_speed = st.text_input('Wind Speed (in meters per second)')
    sky_cover = st.text_input('Sky Cover')
    visibility = st.text_input('Visibility (in kilometers)')
    humidity = st.text_input('Humidity (in percentage)')
    average_wind_speed = st.text_input('Average Wind Speed (in meters per second)')
    average_pressure = st.text_input('Average Pressure (in mercury inches)')
    
    # Variable to store the prediction result
    result_generation = ''
    
    # Create a button for prediction
    if st.button('Solar Power Generation Result'):
        # Convert inputs to float
        try:
            input_features = [
                float(distance_to_solar_noon),
                float(temperature),
                float(wind_direction),
                float(wind_speed),
                float(sky_cover),
                float(visibility),
                float(humidity),
                float(average_wind_speed),
                float(average_pressure)
            ]
            result_generation = solarpower_generation_prediction(input_features)
            st.success(f'Predicted Solar Power Generation: {result_generation}')
        except ValueError:
            st.error('Please enter valid numerical values for all input fields.')
    
if __name__ == '__main__':
    main()
