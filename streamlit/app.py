import streamlit as st




st.set_page_config(
    page_title='Проект: предсказание цен на недвижимость'
)

st.title('Проект: предсказание цен на недвижимость')
st.subheader('Этапы проекта')
st.write('1. Exploratory data analysis (EDA) и работа с выбросами')
st.write('2. Препроцессинг данных и подготовка baseline модели')
st.write('3. Тюнинг модели')


st.sidebar.success('Выбор страницы')