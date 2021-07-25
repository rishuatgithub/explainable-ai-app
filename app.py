import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title='Explainable AI - Heart Disease Prediction')

header = st.beta_container()
dataset = st.beta_container()
dataviz = st.beta_container()
features = st.beta_container()
model = st.beta_container()
explainable = st.beta_container()


@st.cache
def read_data(): 
    data = pd.read_csv('data/heart.csv')
    return data

with header:
    st.title("Explaining Heart Diseases ML Model")
    st.markdown("Many people say machine learning models are **black boxes**, in the sense that they can make good predictions but you can't understand the logic behind those predictions. This statement is true in the sense that most data scientists don't know how to extract insights from models yet.")
    st.markdown("This web interface explains the heart prediction data from [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) dataset using **Explainable AI** technique.")


with dataset:
    
    st.header("Dataset")
    st.markdown("This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.\nIn particular, the Cleveland database is the only one that has been used by ML researchers to this date.\nThe **target** attribute refers to the presence of heart disease in the person.")

    df = read_data()

    dataset_col1, dataset_col2 = st.beta_columns((2,1))
    
    dataset_col1.write(df.head())
    with dataset_col2.beta_expander('Features Description'):
        st.markdown("* **age**: The Age of the person")
        st.markdown("* **sex**: The sex of the person (Male/Female)")
        st.markdown("* **cp**: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)")
        st.markdown("* **thresbps**: The person's resting blood pressure (mm Hg on admission to the hospital)")
        st.markdown("* **chol**: The person's cholesterol measurement in mg/dl")
        st.markdown("* **fbs**: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)")
        st.markdown("* **restecg**: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)")
        st.markdown("* **thalch**: The person's maximum heart rate achieved.")
        st.markdown("* **exang**: Exercise induced angina (1 = yes; 0 = no)")
        st.markdown("* **oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.)")
        st.markdown("* **slope**: The slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping) ")
        st.markdown("* **ca**: Number of major vessels (0-3) colored by flourosopy ")
        st.markdown("* **thal**: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)")
        st.markdown("* **target**: The final prediction value of either 0 (Not Heart Disease) or 1 (Heart Disease)")
    
    dataset_col2.subheader("Total count of rows: ")
    dataset_col2.write(df['age'].count())

with dataviz:
    st.header("Visualizing the dataset")
    st.markdown("Before building the model, let us visualize the above dataset to understand the distribution a little better. ")

    features_list = df[['age','sex','cp']].columns
    selected_feature = st.selectbox('Feature Attributes',features_list)

    dataviz_col1, dataviz_col2 = st.beta_columns(2) 

    barviz_df = pd.DataFrame(df[[selected_feature,'target']].value_counts(), columns=['value'])
    barviz_df.reset_index(level=[selected_feature,'target'], inplace=True)
    barviz_df['target'] = barviz_df['target'].map({0:'Without Heart Disease', 1:'With Heart Disease'})

    fig = px.bar(barviz_df, y=selected_feature, facet_col="target", barmode="group")
    
    dataviz_col1.plotly_chart(fig)


with features: 
    st.header("Feature Engineering")

with model:
    st.header("Model Training and Accuracy")


with explainable:
    st.header("Explaining the Model")