import streamlit as st
import random
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

with open('LE_mdl_v1.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('Ran_mdl.pkl', 'rb') as f: 
    model = pickle.load(f)

def preprocess_input(df):
    df_obj = df.select_dtypes(include=['object'])
    for col, encoder in encoders.items():
        try:
            df[col] = encoder.transform(df[col])
        except:
            df[col] = random.randint(1, 250)
    return df

def main():
    st.title('Fraud Detection Model Prediction')

    st.write('Enter data for prediction:')
    input_data = {}

    features_to_display = ['Credit Card Number (180017442990269)','Merchant name (Rajesh)', 'Category (entertainment, gas_transport)', 'Amount (in US dollor)', 'Gender (M, F)', 'Latitude (Eg 44.2378)', 'Longitude (Eg -95.2739)', 'City Population (Eg 1507)', 'Job (Eg Teacher)', 'Unix Time (Eg 1384969991)', 'Merchant latitude', 'Merchant longitude']
    
    features = ['cc_num','merchant', 'category', 'amt', 'gender', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long']

    for i in range(len(features)):
        input_data[features[i]] = st.text_input(features_to_display[i])

    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        input_df = preprocess_input(input_df)
        input_df.insert(0, 'Unnamed: 0', 388942)
        prediction = model.predict(input_df)
        st.write('Prediction:', "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction")

if __name__ == "__main__":
    main()
