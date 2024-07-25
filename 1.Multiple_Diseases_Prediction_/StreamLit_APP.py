import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd




# Loading the Saved Models
file_path1 = r"C:\ML_PRojects_\Multiple_Diseases_Prediction_\Models\trained_model_diabetes.sav"
with open(file_path1, 'rb') as file:
    diabetes_model = pickle.load(file)

file_path2 = r"C:\ML_PRojects_\Multiple_Diseases_Prediction_\Models\trained_model_Heart_Disease_Prediction.sav"
with open(file_path2, 'rb') as file:
    heart_disease_model = pickle.load(file)

file_path3 = r"C:\ML_PRojects_\Multiple_Diseases_Prediction_\Models\trained_model_Parkinson.sav"
with open(file_path3, 'rb') as file:
    parkinson_disease_model = pickle.load(file)


file_path4 = r"C:\ML_PRojects_\Multiple_Diseases_Prediction_\Models\scaler_parkinsons.pkl"
with open(file_path4,'rb') as file:
    scaler_parkinson = pickle.load(file)


#Sidebar for navigate
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Disease Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Disease Prediction'],
                            icons = ['activity','heart','person'],
                            default_index=0
                            )
    

st.title("Multiple Disease Prediction Using Machine Learning")  

#Diabetes Prediction Page
if (selected == 'Diabetes Disease Prediction'):
    ### page title
    st.title("Diabetes Disease Prediction Using ML")
     
     ## getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    #Code for the prediction
    diab_prediction = ''
    if st.button('Diabetes Test Predict'):
        diab_prediction = diabetes_model.predict(
            [[Pregnancies,Glucose,BloodPressure,
                SkinThickness,Insulin,BMI,
                DiabetesPedigreeFunction ,
                Age]])
        
        if(diab_prediction == [0]):
            diab_prediction = "The Perosn is Diabetic"
        else:
            diab_prediction = "The Person is Non - Diabetic"    

    
    st.success(diab_prediction)






##############################################
#Heart diseas prediction

if(selected == 'Heart Disease Prediction'):
    st.title("Heart Disease Prediction Using ML")#Page Title 

    #Getting the input data fromm user
    
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved') 
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    ca = st.text_input('Major vessels colored by flourosopy')
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        #it hs the float values
        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
   






#############################################################################
#Pasrkions's Prediction Page--
if(selected =='Parkinsons Disease Prediction'):
    st.title("Parkinsons Disease Prediction Using ML") #Page Title
    #Taking the input form the user
    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    Jitter_percent = st.text_input('MDVP:Jitter(%)')
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    RAP = st.text_input('MDVP:RAP')
    PPQ = st.text_input('MDVP:PPQ')
    DDP = st.text_input('Jitter:DDP')
    Shimmer = st.text_input('MDVP:Shimmer')
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    APQ3 = st.text_input('Shimmer:APQ3')
    APQ5 = st.text_input('Shimmer:APQ5')
    APQ = st.text_input('MDVP:APQ')
    DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE') 


    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

     user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

     user_input = [float(x) for x in user_input]
     
     
     #  changing input data to a numpy array
     input_data_as_numpy_array = np.asarray([user_input])##Pass it as 2D array
     
     ### reshape the numpy array
     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

     

     input_scaled_data = scaler_parkinson.transform(input_data_reshaped)

     parkinsons_prediction = parkinson_disease_model.predict(input_scaled_data)
     

        
     if parkinsons_prediction[0] == 1:
        parkinsons_diagnosis = "The person has Parkinson's disease"
     else:
        parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)




 
   





