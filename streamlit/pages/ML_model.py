import joblib
import pandas as pd
import streamlit as st
import sklearn
sklearn.set_config(transform_output="pandas")

st.title('Модель предсказания цен на недвижимость')

ml_model = joblib.load('ml_pipeline_house.pkl')


st.write("Загрузите ваш файл")
uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])

if uploaded_test is not None:
    test = pd.read_csv(uploaded_test)
    predictions = ml_model.predict(test)
    st.write("Предсказанные цены на недвижимость:")
    st.table(predictions)