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

# --- Cargar datos y modelo ---
@st.cache_resource
def cargar_modelo():
    with open("modelo_gasolina.pkl", "rb") as f:
        modelo_data = pickle.load(f)
    return modelo_data

@st.cache_data
def cargar_datos():
    return pd.read_csv("Gasolina_expandido.csv")

modelo_data = cargar_modelo()
df_gasolina = cargar_datos()

# --- Entradas del usuario ---
st.header("Datos de entrada")

# Selectbox con entidades desde el dataset
entidades = sorted(df_gasolina['Entidad'].unique())
entidad = st.selectbox("Entidad Federativa:", entidades)

mes = st.selectbox("Mes:", 
                   ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
                    'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'])

años = sorted(df_gasolina['Año'].unique())
año = st.selectbox("Año:", list(range(min(años), max(años)+5)), index=len(años)-1)


    if st.button("🙌 Predecir"):
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


