# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Previsor de Tsunami - IA",
    page_icon="🌊",
    layout="centered"
)

# Carregar modelo
@st.cache_resource
def load_model():
    return joblib.load('best_tsunami_model.pkl')

model = load_model()

# Título
st.title("Previsor de Tsunami com Inteligência Artificial")
st.markdown("### Baseado em 782 terremotos históricos (2001–2022)")


# Entrada de dados
col1, col2 = st.columns(2)

with col1:
    magnitude = st.slider("Magnitude (Richter)", 5.0, 9.5, 7.0, 0.1)
    sig = st.slider("Significance (sig)", 100, 2000, 800, 50)
    depth = st.slider("Profundidade (km)", 0.0, 700.0, 30.0, 5.0)

with col2:
    latitude = st.number_input("Latitude", -90.0, 90.0, 0.0, 0.5)
    longitude = st.number_input("Longitude", -180.0, 180.0, 120.0, 0.5)

# Botão de previsão
if st.button("Avaliar Risco de Tsunami", type="primary", use_container_width=True):
    dados = pd.DataFrame({
        'magnitude': [magnitude],
        'sig': [sig],
        'depth': [depth],
        'latitude': [latitude],
        'longitude': [longitude]
    })
    
    pred = model.predict(dados)[0]
    prob = model.predict_proba(dados)[0][1]  # probabilidade da classe 1 (tsunami)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.error("ALTO RISCO DE TSUNAMI!")
        st.warning(f"Probabilidade estimada: **{prob:.1%}**")
    else:
        st.success("Baixo risco de tsunami")
        st.info(f"Probabilidade estimada: **{prob:.1%}**")

    st.balloons()

# Rodapé
st.markdown("---")
st.caption("Modelo: Random Forest | F1-Score: ~93% | Features: magnitude, sig, depth, latitude, longitude")