import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart Disease Prediction App
""")

st.sidebar.header('User Input Features')

General_Health = st.sidebar.selectbox(
    'General_Health', ('Poor', 'Very Good', 'Good', 'Fair', 'Excellent'))
Checkup = st.sidebar.selectbox('Checkup', ('Within the past 2 years', 'Within the past year', '5 or more years ago',
                                           'Within the past 5 years', 'Never'))
Exercise = st.sidebar.selectbox('Exercise', ('No', 'Yes'))
Skin_Cancer = st.sidebar.selectbox('Skin_Cancer', ('No', 'Yes'))
Other_Cancer = st.sidebar.selectbox('Other_Cancer', ('No', 'Yes'))
Depression = st.sidebar.selectbox('Depression', ('No', 'Yes'))
Diabetes = st.sidebar.selectbox('Diabetes', ('No', 'Yes', 'No, pre-diabetes or borderline diabetes',
                                             'Yes, but female told only during pregnancy'))
Arthritis = st.sidebar.select_slider('Arthritis', ('No', 'Yes'))
Sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
Age_Category = st.sidebar.selectbox('Age_Category', ('70-74', '60-64', '75-79', '80+', '65-69', '50-54', '45-49', '18-24', '30-34',
                                                     '55-59', '35-39', '40-44', '25-29'))
BMI = st.sidebar.slider('BMI', 10, 60, 100)
Smoking_History = st.sidebar.selectbox('Smoking_History', ('Yes', 'No'))
Alcohol_Consumption = st.sidebar.slider('Alcohol_Consumption', 0.0, 20.0, 40.0)
Fruit_Consumption = st.sidebar.slider('Fruit_Consumption', 0.0, 60.0, 120.0)
Green_Vegetables_Consumption = st.sidebar.slider(
    'Green_Vegetables_Consumption', 0.0, 70.0, 140.0)
FriedPotato_Consumption = st.sidebar.slider(
    'FriedPotato_Consumption', 0.0, 70.0, 140.0)

data = {
    'General_Health': General_Health,
    'Checkup': Checkup,
    'Exercise': Exercise,
    'Skin_Cancer': Skin_Cancer,
    'Other_Cancer': Other_Cancer,
    'Depression': Depression,
    'Diabetes': Diabetes,
    'Arthritis': Arthritis,
    'Sex': Sex,
    'Age_Category': Age_Category,
    'BMI': BMI,
    'Smoking_History': Smoking_History,
    'Alcohol_Consumption': Alcohol_Consumption,
    'Fruit_Consumption': Fruit_Consumption,
    'Green_Vegetables_Consumption': Green_Vegetables_Consumption,
    'FriedPotato_Consumption': FriedPotato_Consumption
}

input_df = pd.DataFrame(data, index=[0])
df_new = pd.read_csv('CVD_newfile.csv')
df_new.drop(columns=df_new.columns[0], axis=1, inplace=True)
df_new.drop('Heart_Disease', axis=1, inplace=True)
df = pd.concat([input_df, df_new], axis=0)

categorical = df.select_dtypes(include=['object']).columns.sort_values()
# categorical=categorical.drop('Heart_Disease')
# df.drop(['Height_(cm)','Weight_(kg)'], axis=1, inplace=True)

for col in categorical:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

st.subheader('User Input features')
st.write(df)

load_model = pickle.load(open('heart_model_1.pkl', 'rb'))

# Apply model to make predictions
preds = load_model.predict(df)
prediction = (preds > 0.5).astype(int)

prediction_proba = load_model.predict_proba(df)

st.subheader('Prediction')
heart_disease = np.array(['No', 'Yes'])
st.write(heart_disease[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
