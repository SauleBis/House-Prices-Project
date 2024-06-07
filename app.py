import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import sklearn
sklearn.set_config(transform_output="pandas")



st.title('Проект: предсказание цен на недвижимости')

st.subheader('Этапы проекта')
st.write('1. Exploratory data analysis (EDA) и работа с выбросами')
st.write('2. Препроцессинг данных и подготовка baseline модели')
st.write('3. Тюнинг модели')

st.subheader('Анализ и подготовка данных к созданию модели')

st.write('Данные для обучения модели')
train = pd.read_csv('Data/train.csv')
st.table(train.head(4))

# График 1
st.subheader('График 1: Зависимость между площадью жилой площади и ценой продажи')
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
st.pyplot(fig)


# График 2
st.subheader('График 2: Зависимость между количеством комнат и ценой продажи')
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(train['TotRmsAbvGrd'], train['SalePrice'], alpha=0.5)
plt.xlabel('Количество комнат')
plt.ylabel('Цена продажи')
plt.title('Зависимость между количеством комнат и ценой продажи')
st.pyplot(fig)

# График 3
st.subheader('График 3: Зависимость между площадью участка и ценой продажи')
filtered_train = train[train['LotArea'] <= 50000]
fig, ax = plt.subplots(figsize=(14, 8))
sns.scatterplot(x='LotArea', y='SalePrice', data=filtered_train, ax=ax)
plt.xticks(rotation=90)
plt.xlabel('Площадь участка')
plt.ylabel('Цена продажи')
plt.title('Зависимость между площадью участка и ценой продажи')
st.pyplot(fig)


# График 4
st.subheader('График 4: Матрица корреляции')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'MasVnrArea', 'YearRemodAdd', 'YearBuilt',
        'GarageYrBlt', 'LotFrontage', 'LotArea', 'BsmtFullBath', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
train_matrix = train[cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(train_matrix, annot=True, cmap='coolwarm', fmt=".1f", ax=ax)
plt.title('Матрица корреляции')
st.pyplot(fig)
st.write('Признаки с высокой мультиколлинеарностью удаляли')

# График 5
st.subheader('График 5: Гистограмма распределения значений')
fig, ax = plt.subplots(figsize=(16, 8))
plt.hist(train['Heating'])
ax.set_title('Гистограмма распределения значений параметра Heating')
st.pyplot(fig)
st.write('Столбцы с признаками, например Heating, не имеющими значения, удаляли полностью')


st.subheader('Pipeline препроцессинга')
st.image('/home/saule/Загрузки/pipeline.png')


st.subheader('Модель предсказания цен на недвижимость')

ml_model = joblib.load('ml_pipeline_house.pkl')


st.write("Загрузите ваш файл")
uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])

if uploaded_test is not None:
    test = pd.read_csv(uploaded_test)
    predictions = ml_model.predict(test)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted Prices'])
    st.write("Предсказанные цены на недвижимость:")
    st.table(predictions_df)
