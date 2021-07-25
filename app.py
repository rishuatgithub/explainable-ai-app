import streamlit as st
import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import preprocessing

from xgboost import XGBClassifier
#import xgboost as xgb

import shap
import eli5
from eli5.sklearn import PermutationImportance

st.set_page_config(layout="wide", page_title='Explaining Heart Diseases ML Model')
st.set_option('deprecation.showPyplotGlobalUse', False)

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

@st.cache
def train_test_split_data(data): 
    X = data.drop("target", 1).values
    y = data["target"].astype("int").values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

    return X_train, X_test, y_train, y_test


with header:
    st.title("Explaining Heart Diseases ML Model")
    #st.markdown("Author: Rishu Shrivastava")
    st.markdown("Many people say machine learning models are **black boxes**, in the sense that they can make good predictions but you can't understand the logic behind those predictions. This statement is true in the sense that most data scientists don't know how to extract insights from models yet.")
    st.markdown("This web interface explains the heart prediction data from [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) dataset using **Explainable AI** technique.")


with dataset:
    
    st.header("Dataset")
    st.markdown("This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.\nIn particular, the Cleveland database is the only one that has been used by ML researchers to this date.\nThe **target** attribute refers to the presence of heart disease in the person.")

    df = read_data()

    dataset_col1, dataset_col2 = st.beta_columns((2,1))
    
    dataset_col1.write(df.head())
    with dataset_col2.beta_expander('Features Description'):
        st.markdown("""
            * **age**: The Age of the person
            * **sex**: The sex of the person (Male/Female)
            * **cp**: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
            * **thresbps**: The person's resting blood pressure (mm Hg on admission to the hospital)
            * **chol**: The person's cholesterol measurement in mg/dl
            * **fbs**: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
            * **restecg**: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
            * **thalch**: The person's maximum heart rate achieved.
            * **exang**: Exercise induced angina (1 = yes; 0 = no)
            * **oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.)
            * **slope**: The slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping) 
            * **ca**: Number of major vessels (0-3) colored by flourosopy 
            * **thal**: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
            * **target**: The final prediction value of either 0 (Not Heart Disease) or 1 (Heart Disease)
        """)
    
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
    st.header("Preparing dataset before training")
    st.markdown("The dataset is having some features with **categorical** dataset. We will apply [dummy encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) to convert the categorical data to binary features.")
    st.markdown("The list of categorical features are: cp, sex, exang, slope, thal, restecg, fbs. ")

    data_catg = df.copy()
    data_catg['cp'] = df['cp'].map({0:"asymptomatic",1:"typical_angina",2:"atypical_angina",3:"non_anginal_pain"})
    data_catg['sex'] = df['sex'].map({0:"female", 1:"male"}) 
    data_catg['exang'] = df['exang'].map({0:"exercise_not_induce_angina", 1:"exercise_induced_angina"})
    data_catg['slope'] = df['slope'].map({1:"upsloping", 2:"flat", 3:"downsloping"})
    data_catg['thal'] = df['thal'].map({1:"normal",2:"fixed_defect", 3:"reversable_defect"})
    data_catg['restecg'] = df['restecg'].map({0:'normal',1:'st_t_wave_abnormal',2:'left_ventricular_hypertrophy'})
    data_catg['fbs'] = df['fbs'].map({0:'lower_than_120mg_per_ml',1:'higher_than_120mg_per_ml'})

    df = pd.get_dummies(data_catg, drop_first = True)

    st.markdown("After applying the dummy encoding scheme, the original dataset now looks like as below. Notice the new encoded features being added.")
    st.write(df.head())

    st.text("Note: There are many feature engineering techniques that could be applied on this dataset. For the purpose of this exercise, we will not deep dive into feature engineering other than small encoding done above.")

with model:
    st.header("Model Training")
    st.markdown("We will use **XGBoost classifier** algorithm to train on the dataset based on the selection of parameters.")

    ## splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    model_col1, model_col2 = st.beta_columns((1,2))

    model_max_depth = model_col1.slider("Max Depth", min_value=3, max_value=10, step=1, value=5)
    model_learning_rate = model_col1.selectbox("Learning Rate", options=[0.001, 0.01, 0.1, 0.3, 0.5], index=3)
    model_estimators = model_col1.selectbox("Number of estimators", options=[100, 300, 500], index=0)

    model = XGBClassifier(max_depth=model_max_depth, learning_rate=model_learning_rate, n_estimators= model_estimators, n_jobs=1)
    model.fit(X_train, y_train)

    yhat = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy_score = accuracy_score(yhat,y_test)
    tp,fn,fp,tn = confusion_matrix(y_test, yhat, labels=[1,0]).ravel()
    precision_rate = tp / (tp + fp)
    recall_rate = tp / (tp + fn)
    
    model_col2.subheader("Model Accuracy")
    model_col2.write(accuracy_score)

    model_col2.subheader("Model Precision")
    model_col2.write(precision_rate)

    model_col2.subheader("Model Recall")
    model_col2.write(recall_rate)


with explainable:
    st.header("Explaining the Model")

    feature_dict = dict(enumerate(df.drop("target", 1).columns))

    st.subheader("Permutation Importance")
    
    #perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
    #permutation_imp_chart = eli5.show_weights(perm, feature_names = list(feature_dict.values()))
    #st.pyplot(permutation_imp_chart)

    
    
    st.subheader("Partial Plots")



    st.subheader("SHAP")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    #summary_plot = shap.summary_plot(shap_values, X_test, feature_names = list(feature_dict.values()), plot_type = "bar")
    summary_plot = shap.summary_plot(shap_values, X_test, feature_names = list(feature_dict.values()))
    st.pyplot(summary_plot)