import streamlit as st
import joblib 
import numpy as np
import pandas as pd

model=joblib.load('startup_rf.pkl')
encoded_columns=joblib.load('encoded_columns.pkl')

st.title("Startup Funding Prediction")

df=pd.read_csv('startup_cleaned.csv')

subvertical_options = df['SubVertical'].unique().tolist()
city_options = df['City'].unique().tolist()
industry_options = df['Industry Vertical Cleaned'].unique().tolist()
investment_type_options = df['InvestmentType_Cleaned'].unique().tolist()

subvertical = st.selectbox('Select SubVertical',subvertical_options)
city = st.selectbox('Select City',city_options)
industry = st.selectbox('Select Industry',industry_options)
investment_type = st.selectbox('Select Investment Type',investment_type_options)
year=st.number_input('Year',min_value=2000,max_value=2025,value=2023)
month=st.number_input('Month',min_value=1,max_value=12,value=6)

def process_input():

    input_df=pd.DataFrame(np.zeros((1,len(encoded_columns))),columns=encoded_columns)

    input_df['Year']=year
    input_df['Month']=month

    col_name = f'SubVertical_{subvertical}'
    if col_name in input_df.columns:
        input_df.loc[0, col_name]=1

    col_name= f'City_{city}'
    if col_name in input_df.columns:
        input_df.loc[0, col_name]=1

    col_name = f'Industry Vertical Cleaned_{industry}'
    if col_name in input_df.columns:
        input_df.loc[0, col_name]=1

    col_name = f'InvestmentType_Cleaned_{investment_type}'
    if col_name in input_df.columns:
        input_df.loc[0, col_name]=1

  


    return input_df

if st.button('Predict Funding'):
    features=process_input()

    if 'is_funded' in features.columns:
        features = features.drop(columns=['is_funded'])
        

    pred=model.predict(features)[0]
    prob=model.predict_proba(features)[0][1]

    if pred == 1:
        st.success(f'Prediction: Funded! Probability: {prob:.2f}')
    else:
        st.error(f'Prediction: Not Funded. Probability: {prob:.2f}')