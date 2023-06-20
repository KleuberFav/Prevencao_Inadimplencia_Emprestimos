import pandas as pd 
import numpy as np
import joblib
import pickle
import streamlit as st


# Carregando o pipeline salvo anteriormente
modelo_carregado = joblib.load('outputs/grid_search.pkl')

# Carregar a lista do arquivo
with open('outputs/lista_vars_numericas.pkl', 'rb') as arquivo:
    lista_vars_numericas = pickle.load(arquivo)

# Carregar a lista do arquivo
with open('outputs/lista_vars_dummif.pkl', 'rb') as arquivo:
    lista_vars_dummif = pickle.load(arquivo)

# Carregar a lista do arquivo
with open('outputs/lista_vars_le.pkl', 'rb') as arquivo:
    lista_vars_le = pickle.load(arquivo)

st.set_page_config(layout="wide")




st.title("Solicitação de Empréstimo")




# Carregar arquivo CSV

col1, col2 = st.columns([2, 3])

with col1:
    file = st.file_uploader("Insira os dados do solicitante")


# Processar o arquivo, se houver
with col2:
    if file is not None:

        df = pd.read_csv(file)
        
        X_novo = df.drop(lista_vars_le, axis=1)
        
        score = modelo_carregado.predict_proba(X_novo)
        score = score[0,0]

        if score > 0.933892:
            st.markdown("<h1 style='font-size: 40px'>Score: {}</h1>".format(round(score,3)), unsafe_allow_html= True)
            st.markdown("<h2 style='font-size: 40px'>Situação: Empréstimo Aprovado</h1>", unsafe_allow_html= True)
        elif 0.838630 < score <= 0.933892:
            st.markdown("<h1 style='font-size: 40px'>Score: {}</h1>".format(round(score,3)), unsafe_allow_html= True)
            st.markdown("<h2 style='font-size: 40px'>Situação: Enviar para um analista</h1>", unsafe_allow_html= True)
        else:
            st.markdown("<h1 style='font-size: 40px'>Score: {}</h1>".format(round(score,3)), unsafe_allow_html= True)
            st.markdown("<h2 style='font-size: 40px'>Situação: Empréstimo Negado</h1>", unsafe_allow_html= True)







