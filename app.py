import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

# Title
st.title("🎓 Student Grade Prediction App")

st.write("Enter student details to predict final grade (G3)")

# Inputs
g1 = st.number_input("G1 (First Period Grade)", min_value=0, max_value=20, value=10)
g2 = st.number_input("G2 (Second Period Grade)", min_value=0, max_value=20, value=10)
studytime = st.number_input("studytime (1-4)", min_value=1, max_value=4, value=1)


# Prediction button
if st.button("Predict Grade"):
    input_data = np.array([[g1, g2, studytime]])
    
    prediction = model.predict(input_data)[0]
    
    # Fix range
    prediction = max(0, min(20, prediction))
    
    st.success(f"Predicted Final Grade (G3): {round(prediction)} / 20")