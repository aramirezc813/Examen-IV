import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("⛽ Predicción de Precios de Gasolina en México")

st.write("""
Esta aplicación predice el **precio de la gasolina** en México a partir de:
- Estado/Entidad  
- Mes  
- Año  
""")

# --- Cargar modelo entrenado ---
@st.cache_resource
def cargar_modelo():
    with open("modelo_gasolina.pkl", "rb") as f:
        modelo_data = pickle.load(f)
    return modelo_data

modelo_data = cargar_modelo()

# --- Entradas del usuario ---
st.header("Datos de entrada")

entidad = st.text_input("Entidad Federativa:", "Nacional")
mes = st.selectbox("Mes:", 
                   ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
                    'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'])
año = st.number_input("Año:", min_value=2000, max_value=2050, value=2025, step=1)

# --- Predicción ---
if st.button("🔮 Predecir"):
    try:
        modelo = modelo_data['modelo']
        encoder_entidad = modelo_data['encoder_entidad']
        encoder_mes = modelo_data['encoder_mes']

        entidad_encoded = encoder_entidad.transform([entidad])[0]
        mes_encoded = encoder_mes.transform([mes])[0]

        X_pred = np.array([[entidad_encoded, mes_encoded, año]])
        precio_predicho = modelo.predict(X_pred)[0]

        st.success(f"💰 Precio estimado: **${precio_predicho:.2f} MXN**")
        st.write(f"({entidad}, {mes} {año})")

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
