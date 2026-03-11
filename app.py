import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Netflix Content Classifier")

text = st.text_area("Enter Description")

if st.button("Predict"):

    text = text.lower()

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)

    st.success(f"Prediction: {prediction[0]}")