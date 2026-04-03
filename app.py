import streamlit as st
import pickle
import numpy as np
import os
import google.generativeai as genai


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  

model_ai = genai.GenerativeModel("gemini-2.5-flash")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "student_model.pkl")

model = pickle.load(open(model_path, "rb"))


def get_ai_recommendation(g1, g2, studytime, prediction):
    prompt = f"""
    A student has:
    - G1 (first grade): {g1} out of 20
    - G2 (second grade): {g2} out of 20
    - Study time level: {studytime} (1=low, 4=high)
    - Predicted final grade: {prediction} out of 20

    Note:
    - 0-10 = low performance
    - 10-15 = average performance
    - 15-20 = good performance

    Give 3 simple, practical and realistic recommendations 
    specifically to improve marks (not generic advice).
    Keep it short and student-friendly.
    """
    
    response = model_ai.generate_content(prompt)
    return response.text
st.title("🎓 Student Grade Prediction App")

st.write("Enter student details to predict final grade (G3)")

g1 = st.number_input("G1 (First Period Grade)", 0, 20, 10)
g2 = st.number_input("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.number_input("Studytime (1-4)", 1, 4, 1)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Grade"):
    input_data = np.array([[g1, g2, studytime]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.success(f"Predicted Grade: {round(prediction)} / 20")

    # 🔥 AI RECOMMENDATIONS (MOVE INSIDE BUTTON)
    ai_response = get_ai_recommendation(g1, g2, studytime, round(prediction))

    st.subheader("🤖 AI Recommendations")
    st.write(ai_response)
