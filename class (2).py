import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = joblib.load('model.pkl')


st.title("Classification Model Deployment")
st.write("Input the features to get predictions from the model.")


age = st.number_input('Age', min_value=0, max_value=100, step=1)
sex = st.selectbox('Sex', ['Male', 'Female']) 
albumin = st.number_input('Albumin', min_value=0.0, step=0.1)
alkaline_phosphatase = st.number_input('Alkaline Phosphatase', min_value=0.0, step=0.1)
alanine_aminotransferase = st.number_input('Alanine Aminotransferase', min_value=0.0, step=0.1)
aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0.0, step=0.1)
bilirubin = st.number_input('Bilirubin', min_value=0.0, step=0.1)
cholinesterase = st.number_input('Cholinesterase', min_value=0.0, step=0.1)
cholesterol = st.number_input('Cholesterol', min_value=0.0, step=0.1)
creatinina = st.number_input('Creatinina', min_value=0.0, step=0.1)
gamma_glutamyl_transferase = st.number_input('Gamma Glutamyl Transferase', min_value=0.0, step=0.1)
protein = st.number_input('Protein', min_value=0.0, step=0.1)

sex_encoded  = 0 if sex == 'Male' else 1 

label_encoder = LabelEncoder()

label_encoder.fit(['no_disease', 'suspect_disease', 'hepatitis', 'fibrosis', 'cirrhosis'])

if st.button('Predict'):
  
    input_data = np.array([[age, sex_encoded, albumin, alkaline_phosphatase,alanine_aminotransferase, aspartate_aminotransferase, 
                            bilirubin, cholinesterase, cholesterol, creatinina, 
                            gamma_glutamyl_transferase, protein]])

   
    prediction = model.predict(input_data)

    mapped_predictions = label_encoder.inverse_transform(prediction)

    st.write("Predicted Categories:")
    for i, category in enumerate(mapped_predictions):
           st.write(f"Prediction {i+1}: {category}")
