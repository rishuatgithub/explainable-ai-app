import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import shap
import eli5
from eli5.sklearn import PermutationImportance

from pdpbox import pdp


st.set_page_config(layout="wide", page_title='Explaining Heart Diseases ML Model')
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.initjs()

header = st.beta_container()
dataset = st.beta_container()
#features = st.beta_container()
model = st.beta_container()
explainable = st.beta_container()

@st.cache
def read_data():
    '''
        Read the dataset
        @return: dataframe with the renamed column names
    '''
    data = pd.read_csv('data/heart.csv')

    data.columns = ['age','sex','chest_pain_agnia','resting_blood_pressure','cholestrol','fasting_blood_sugar',
        'resting_ecg','max_heart_rate','exercise_induced_agnia','st_depression_rt_rest','slope','number_of_major_vessels','thalassemia','target']

    return data

@st.cache
def train_test_split_data(df):
    '''
        One hot encode the dataframe and return the train/test split
    '''

    data_catg = df.copy()
    data_catg['chest_pain_agnia'] = df['chest_pain_agnia'].map({0:"asymptomatic",1:"typical",2:"atypical",3:"non_anginal"})
    data_catg['sex'] = df['sex'].map({0:"female", 1:"male"}) 
    data_catg['exercise_induced_agnia'] = df['exercise_induced_agnia'].map({0:"false", 1:"true"})
    data_catg['slope'] = df['slope'].map({1:"upsloping", 2:"flat", 3:"downsloping"})
    data_catg['thalassemia'] = df['thalassemia'].map({1:"normal",2:"fixed_defect", 3:"reversable_defect"})
    data_catg['resting_ecg'] = df['resting_ecg'].map({0:'normal',1:'st_wave_abnormal',2:'left_ventricular_hypertrophy'})
    data_catg['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({0:'<=120mg/ml',1:'>120mg/ml'})

    df = pd.get_dummies(data_catg, drop_first = True)

    X = df.drop("target", 1).values
    y = df["target"].astype("float").values

    ## column is used for the charts below
    encoded_df_column_list = df.columns.drop('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

    return X_train, X_test, y_train, y_test, encoded_df_column_list


with header:
    st.title("Explaining Heart Diseases ML Model")
    #st.markdown("Author: Rishu Shrivastava")
    st.markdown("""
        Many people say machine learning models are **black boxes**, in the sense that they can make good predictions but you can't understand the logic behind those predictions. This statement is true in the sense that most data scientists don't know how to extract insights from models yet.")
        
        This interactive application explains the presence of heart disease in a person based on [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) dataset using **Explainable AI** technique.
    """)


with dataset:

    st.header("**Dataset**")
    st.markdown("""
        This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.
        In particular, the Cleveland database is the only one that has been used by ML researchers to this date.
        The **target** attribute refers to the presence of heart disease in the person.
    """)

    df = read_data()

    st.write(df.head())

    #dataset_col2.subheader("Total count of rows: ")
    #dataset_col2.write(df['age'].count())

    #with dataset_col2.beta_expander('Features Description'):
    #    st.markdown("""
    #        * **age**: The Age of the person
    #        * **sex**: The sex of the person (Male/Female)
    #        * **cp**: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
    #        * **thresbps**: The person's resting blood pressure (mm Hg on admission to the hospital)
    #        * **chol**: The person's cholesterol measurement in mg/dl
    #        * **fbs**: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
    #        * **restecg**: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
    #        * **thalch**: The person's maximum heart rate achieved.
    #        * **exang**: Exercise induced angina (1 = yes; 0 = no)
    #        * **oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.)
    #        * **slope**: The slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping) 
    #        * **ca**: Number of major vessels (0-3) colored by flourosopy 
    #        * **thal**: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
    #        * **target**: The final prediction value of either 0 (Not Heart Disease) or 1 (Heart Disease)
    #    """)
    
    st.markdown("""
        The dataset is having some features with **categorical** dataset. 
        We will apply [dummy encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) technique to convert the categorical data to binary features.
    """)
    st.text("""
        Note: There are many feature engineering techniques that could be applied on this dataset. 
              For the purpose of this exercise, we will not deep dive into feature engineering other than encoding done above.
        """)

with model:
    st.header("**Model Training**")
    st.markdown("""
        We will use **XGBoost classifier** algorithm to train on the dataset based on the selection of parameters.
        The dataset uses 80:20 train-test split ratio of data.
    """)

    ## splitting the dataset
    X_train, X_test, y_train, y_test, encoded_df_column_list = train_test_split_data(df)

    model_col1, model_col2, model_col3 = st.beta_columns((1,1,2))

    model_max_depth = model_col1.slider("Max Depth", min_value=3, max_value=10, step=1, value=5)
    model_learning_rate = model_col1.selectbox("Learning Rate", options=[0.001, 0.01, 0.1, 0.3, 0.5], index=1)
    model_estimators = model_col1.selectbox("Number of estimators", options=[100, 300, 400, 500, 700], index=3)

    model = XGBClassifier(max_depth=model_max_depth, 
                        learning_rate=model_learning_rate, 
                        n_estimators= model_estimators, 
                        use_label_encoder=False,
                        n_jobs=1)

    eval_set = [(X_train, y_train), (X_test, y_test)]

    model_train = model.fit(X_train, y_train, eval_metric=['error','logloss'], eval_set=eval_set, verbose=False)
    
    y_pred = model_train.predict(X_test)
    y_proba = model_train.predict_proba(X_test)[:, 1]

    accuracy_score = accuracy_score(y_pred,y_test)
    tp,fn,fp,tn = confusion_matrix(y_test, y_pred, labels=[1,0]).ravel()
    precision_rate = tp / (tp + fp)
    recall_rate = tp / (tp + fn)

    #F1 = 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision_rate * recall_rate) / (precision_rate + recall_rate)

    results = model_train.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    
    model_col2.subheader("**Model Accuracy**")
    model_col2.write(round(accuracy_score*100,2))

    model_col2.subheader("**Model Precision**")
    model_col2.write(round(precision_rate*100,2))

    model_col2.subheader("**Model Recall**")
    model_col2.write(round(recall_rate*100,2))

    model_col2.subheader("**Model F1 Score**")
    model_col2.write(round(f1_score*100,2))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(x_axis), y=results['validation_0']['logloss'], name='Train'))
    fig.add_trace(go.Scatter(x=list(x_axis), y=results['validation_1']['logloss'], name='Test'))
    fig.update_layout(title='<b>Model Loss</b>',
                margin=dict(l=1,r=1,b=0), 
                height=400, 
                width=600,
                xaxis_title="Epochs", 
                yaxis_title="Model Loss")

    model_chart_error_loss = fig

    model_col3.plotly_chart(model_chart_error_loss)

    st.text("_Note: All the prediction values are displayed in percentage (%)")


with explainable:

    st.header("**Explaining the Model**")
    st.markdown("""
        In the above section, XGBoost model predicted some output score based on the heart disease prediction dataset. 
        Based on the initial model parameters, a **F1 score** of `"""+str(round(f1_score*100,2))+"""%` was acheived based on the validation/test data.
        However, there are open questions that would easily come to mind:

        1. What features in the data did the model **think are most important**?
        2. How did each **feature** in the data **affect a particular prediction**?
        3. How does each feature **affect** the model's predictions **over a larger dataset**?

        In the field of **Explainable AI**, by definition, [explainability](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) is considered as "the collection of features of the interpretable domain, that have contributed for a given example to produce a decision (e.g., classification or regression)".
        If algorithms meet these requirements, they provide a basis for justifying decisions, tracking and thereby verifying them, improving the algorithms, and exploring new facts.

        Based on the above trained model, let us try to answer the above three basic questions.
        """)
    
    feature_dict = dict(enumerate(encoded_df_column_list))
    features_list = encoded_df_column_list
    
    st.markdown("### **Permutation Importance**")

    pi_col1, pi_col2 = st.beta_columns(2)

    pi_col1.markdown("""
        ** _What are the features that have the biggest impact on the prediction?_ **

        This concept of finding the feature importance is called Permutation Importance. This technique is fast to calculate and easy to understand. 
        The feature imporatance is calculated based on the trained model. 

        The values on the top are the most important features, and those at the bottom are the least. 
        On the our dataset, _(based on the initial model parameters)_ the top 3 most important features are `number_of_major_vessel`, `cholestrol` and `st_depression_rt_rest`. 
        
        Model thinks that the presence of blood diseases like thalessemmia, higher cholestrol levels in a persons are some of the key reasons for having a heart disease.
        According to the [NHS - UK website](https://www.nhs.uk/conditions/cardiovascular-disease/), heart related diseases are caused by having some pre-conditions in the blood of a person.
        
        
    """)
    
    perm = PermutationImportance(model_train, random_state=1).fit(X_test, y_test)
    permutation_imp_chart = eli5.show_weights(perm, feature_names = list(feature_dict.values())).data 
    pi_col2.markdown(permutation_imp_chart.replace('\n',''),unsafe_allow_html=True)

    
    
    st.markdown("### **Partial Dependence Plots**")

    pp_col1, pp_col2 = st.beta_columns(2)

    selected_feature = pp_col2.selectbox('Feature Attributes',features_list, index=5)

    pp_col1.markdown("""
        ** _How a feature effects a prediction?_ **

        Partial Dependence Plots (PDP) show the dependence between the target response and a set of input features of interest, 
        marginalizing over the values of all other input features (the ‘complement’ features). 
        Intuitively, we can interpret the partial dependence as the expected target response as a function of the input features of interest.
        PDP are also used in Google's [What-If Tool](https://pair-code.github.io/what-if-tool/learn/tutorials/walkthrough/).
        The target here is to predict the heart related diseases.

        In PDPs, the y-axis or the feature column predicts the **change in prediction** from what it would be predicted at the baseline or left-most value.

        Let's look into one of the feature: `number_of_major_vessels`. With the increase in the number of vessels, the model thinks 
        the probability of having a heart diseases decreases.

        For feature: `max_heart_rate` the chance of having a heart disease increases with the increase in the heart rate.

    
    """)

    X_test_df = pd.DataFrame(X_test).rename(columns=feature_dict)

    fig = plt.figure()
    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test_df, model_features=features_list, feature=selected_feature)
    pdp_dist = pdp.pdp_plot(pdp_dist, selected_feature)
    pdp_dist = plt.show()
    pp_col2.pyplot(pdp_dist, bbox_inches='tight')


    st.markdown("### **SHAP (SHapley Additive exPlanations) **")

    st.markdown("""
        **_How much a prediction was driven by the fact that a person's_ `max_heart_rate` _is greater than 120?_**

        A prediction can be explained by assuming that each feature value of the instance is a “player” in a game where the prediction is the payout. 
        [Shapley values](https://christophm.github.io/interpretable-ml-book/shapley.html) – a method from coalitional game theory – tells us how to fairly distribute the “payout” among the features.
        SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

        The SHAP values provide two great advantages:

        - **Global interpretability**: The SHAP values can show how much each predictor contributes, either positively or negatively, to the target variable. 
            This is like the partial dependence plot but it is able to show the positive or negative relationship for each variable with the target.
        - **Local interpretability**: Each observation gets its own set of SHAP values. This greatly increases its transparency. 
                                We can explain why a case receives its prediction and the contributions of the predictors. 
                                Traditional variable importance algorithms only show the results across the entire population but not on each individual case. 
                                The local interpretability enables us to pinpoint and contrast the impacts of the factors.

    
    """)
    shap_col1, shap_col2 = st.beta_columns(2)

    shap_col1.markdown("""
        The chart on the right hand side shows the indivisual feature contribution towards predicting the model's output.

        If you select a `Person: 1` from the selection box, you could see a series of blue and red features contributing towards predicting the 
        presenece of heart disease in that person. Based on the initial parameters of this model, the presence of heart disease is very low. 
    
    """)

    select_person = shap_col2.selectbox('Select the Person',range(1,len(X_test)),index=1)
    
    ## shap person plot
    select_person_row = pd.DataFrame(X_test).rename(columns=feature_dict).iloc[[select_person]]

    def plot_force_shap_values(model, patient_data):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_data)
        shap.initjs()
        return shap.force_plot(explainer.expected_value, 
                                shap_values, 
                                patient_data, 
                                matplotlib=True, 
                                show=False, 
                                feature_names = list(feature_dict.values()), 
                                text_rotation=10)

    shap_plt = plot_force_shap_values(model, select_person_row.values)
    shap_col2.pyplot(shap_plt)
    plt.clf()

    
    ## summary plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    summary_plot = shap.summary_plot(shap_values, X_test, feature_names = list(feature_dict.values()))
    shap_col2.pyplot(summary_plot)


    