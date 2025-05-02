import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.joblib")
st.title("Clasificador de Flores Iris")

sl = st.slider("Largo del sépalo (cm)", 4.0, 8.0, 5.0)
sw = st.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.0)
pl = st.slider("Largo del pétalo (cm)", 1.0, 7.0, 4.0)
pw = st.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.0)

if st.button("Predecir"):
    data = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(data)
    clases = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Tipo de flor: {clases[prediction[0]]}")
