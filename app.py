import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import openpyxl

st.set_page_config(
    page_title="AppPredvit",
    layout="wide"
)

def load_data_and_model():
    df = pd.read_excel("vitimas.xlsx")
    encoder = OrdinalEncoder()

    for col in df.columns.drop('qtdRegAnt'):
        df[col] = df[col].astype('category')
        X_encoded = encoder.fit_transform(df.drop('qtdRegAnt', axis=1))

        y = df['qtdRegAnt'].astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

        modelo = LogisticRegression(C=0.806, class_weight='balanced', dual=False,
                                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                                    max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2',
                                    random_state=2837, solver='lbfgs', tol=0.0001, verbose=0,
                                    warm_start=False)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        precisao = precision_score(y_test, y_pred)
        revocacao = recall_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        return encoder, modelo, df, acuracia, precisao, revocacao, f_score, auc
    
encoder, modelo, df, acuracia, precisao, revocacao, f_score, auc = load_data_and_model()

st.title("Previsão de Reincidência como Vítima em Ocorrência de Violência Doméstica")

input_features = [
        st.selectbox("A vítima possui filhos com o autor?", df['filhosJuntos'].unique()),
        st.selectbox("Estão juntos a quanto tempo?", df['tempoJuntos'].unique()),
        st.selectbox("Qual a faixa de idade da vítima?", df['idade'].unique()),
        st.selectbox("A vítima possui um trabalho?", df['possuiProfissao'].unique()),
        st.selectbox("Qual o nível de escolaridade da vítima?", df['escolaridade'].unique()),
        ]

if st.button("Prever"):
    input_df = pd.DataFrame([input_features], columns=df.columns.drop('qtdRegAnt'))
    input_encoded = encoder.transform(input_df)
    predict_encoded = modelo.predict(input_encoded)
    previsao = df['qtdRegAnt'].astype('category').cat.categories[predict_encoded][0]
    st.header(f"Resultado da previsão: {previsao}")

st.header("Ficha técnica do modelo")
df1 = pd.DataFrame(
    data = [[{acuracia},{precisao},{revocacao},{f_score},{auc}]],
    columns=['Acurácia','Precisão','Revocação','F1_score','AUC'],

)
st.dataframe(df1,hide_index=True)
