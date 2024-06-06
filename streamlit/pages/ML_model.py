import joblib
import pandas as pd
import streamlit as st

st.title('Модель предсказания цен на недвижимость')

st.write("Загрузите ваш файл")
uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])

if uploaded_test is not None:
    test = pd.read_csv(uploaded_test)
    st.write("Предсказанные цены на недвижимость:")
    st.table(predictions)