# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:45:20 2024

@author: user
"""
import numpy as np
import pickle
import streamlit as st

#loading the model
loaded_model = pickle.load(open('C:/Users/user/Desktop/ML Deployment/trained_model.sav','rb'))

#creating a function
def diabetes_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0]==1:
      return 'Has Diabetes'
    else:
      return "You are good!"

def main():
    #giving a title to the page
    st.title('Diabetes Predicting System')    
    
    #getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucouse = st.text_input('Glucouse Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age')

    #code for prediction
    diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucouse,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]) 
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()    