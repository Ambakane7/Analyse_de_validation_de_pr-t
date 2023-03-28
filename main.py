import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.markdown("<h2 style ='text-align:center; color: green;'>Application de Machine Learning pour la Validation de credit immobilier du Dataset 'Home Loan' de Kaggle</h2>", unsafe_allow_html=True)
st.markdown("<h4 style ='text-align:center;'>Auteur : Ambakane Guindo dit Mr_G</h>", unsafe_allow_html=True)

#####################################
# affichage 

    
@st.cache_data(persist=True)
def load_data():
        data= pd.read_csv('test.csv')
        return data
    # affichage de la table de donnée
df = load_data()
df_sample = df.sample(5)
if st.sidebar.checkbox('Afficher les données brutes',False):
        st.subheader("Dataset de 100 observations")
        st.write(
        df_sample.style.set_properties(**{'background-color': 'lightgray', 'color': 'black'}).set_precision(2)
        )

#Collecter le profil d'entrée
st.sidebar.header("Les caracteristiques du client")

def client_caract_entree():
    Gender=st.sidebar.selectbox('Sexe',('1','0'))
    Married=st.sidebar.selectbox('Marié',('1','0'))
    Dependents=st.sidebar.selectbox('Enfants',('0','1','2','3'))
    Education=st.sidebar.selectbox('Education',('1','0'))
    Self_Employed=st.sidebar.selectbox('Salarié ou Entrepreneur',('1','0'))
    ApplicantIncome=st.sidebar.slider('Salaire du client',150,4000,200)
    CoapplicantIncome=st.sidebar.slider('Salaire du conjoint',0,40000,2000)
    LoanAmount=st.sidebar.slider('Montant du crédit en Kdollar',9.0,700.0,200.0)
    Loan_Amount_Term=st.sidebar.selectbox('Durée du crédit',(360.0,120.0,240.0,180.0,60.0,300.0,36.0,84.0,12.0))
    Credit_History=st.sidebar.selectbox('Credit_History',(1.0,0.0))
    Property_Area=st.sidebar.selectbox('Property_Area',('1','0','2'))

    data={
    'Gender':Gender,
    'Married':Married,
    'Dependents':Dependents,
    'Education':Education,
    'Self_Employed':Self_Employed,
    'ApplicantIncome':ApplicantIncome,
    'CoapplicantIncome':CoapplicantIncome,
    'LoanAmount':LoanAmount,
    'Loan_Amount_Term':Loan_Amount_Term,
    'Credit_History':Credit_History,
    'Property_Area':Property_Area
    }

    profil_client=pd.DataFrame(data,index=[0])
    return profil_client

input_df=client_caract_entree()

df=pd.read_csv('test.csv')
df['Education'].replace(['Graduate', 'Not Graduate'], ['Graduate','Not_Graduate'], inplace=True)
credit_input=df.drop(columns=['Loan_ID'])
donnee_entree=pd.concat([input_df,credit_input],axis=0)


##############################################
code = {
    'Male':0,
    'Female':1,
    'Yes':1,
    'No':0,
    'Rural':0,
    'Urban':1,
    'Semiurban':2,
    "1":1,
    'Graduate':1,
    'Not_Graduate':0,
    '0':0,
    '2':2,
    'Y':1,
    'N':0,
    '3+':3,
           }

for col in df.select_dtypes('object'):
  df[col]=df[col].map(code)
  
  
donnee_entree=donnee_entree[:1]

st.subheader('Les caracteristiques transformés')
st.write(donnee_entree)

################################################
#importer le modèle
load_model=pickle.load(open('prevision_credit.pkl','rb'))


#appliquer le modèle sur le profil d'entrée
prevision=load_model.predict(donnee_entree)

st.subheader('Résultat de la prévision')
st.write(prevision)
if prevision ==1:
    st.write('Le sera Prêt Accordé')
else:
    st.write('Le Prêt ne sera pas Accordé')
    
    
st.write("Notre prédiction se base la décision de l'algorithme de RandomForest que nous avons jugé adéquat pour sa précision et aussi le peu d'erreur commis")
st.write("<h6 style='color: red;'>Adresser vous au developpeur pour une meilleur comprehension de notre modèl, à savoir qu'il pourra être amélioré dans les jours à vénir</h6>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)