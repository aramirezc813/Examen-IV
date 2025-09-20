import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("⛽Predicción de Precios de Gasolina en México")

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


# --- Predicción ---
col1, col2, col3 = st.columns([1,3,1])

with col2:  # ahora sí centrado
    if st.button("🔮 Predecir"):
        try:
            modelo = modelo_data['modelo']
            encoder_entidad = modelo_data['encoder_entidad']
            encoder_mes = modelo_data['encoder_mes']

            entidad_encoded = encoder_entidad.transform([entidad])[0]
            mes_encoded = encoder_mes.transform([mes])[0]

            X_pred = np.array([[entidad_encoded, mes_encoded, año]])
            precio_predicho = modelo.predict(X_pred)[0]

            # Bloque con degradado rosa
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                font-size: 1.5rem;
                font-weight: bold;
                margin-top: 1rem;
            ">
                💰 ${precio_predicho:.2f} MXN  
                <br>
                <span style="font-size:1rem; font-weight:normal;">
                    {entidad} • {mes} {año}
                </span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")

