# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:05:35 2023

@author: ADMIN
"""
import pandas as pd
import streamlit as st
import pickle
import math
model = pickle.load(open('reg.pkl', 'rb'))

#Caching the model for faster loading
@st.cache_data
def predict(mileage,age):
    prediction = model.predict(pd.DataFrame([[mileage,age]],columns=['mileage','age']))
    output = math.floor(prediction)
    if output<0:
        return 'Sorry you cannot sell your bike' 
    else :
    
        return float(output)


st.title('Bike price prediction')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.header('Enter the mileage and age of bike')
mileage = st.number_input('Mileage', min_value=0.1, max_value=1000000000.0, value=1.0)
age =  st.number_input('Age', min_value=0.1, max_value=100.0, value=1.0)      
        
if st.button('Predict'):
    price = predict(mileage,age)
    st.success(price)