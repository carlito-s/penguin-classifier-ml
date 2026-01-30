import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="IA Pingüinos", page_icon="🐧")

@st.cache_resource
def load_model():
    return joblib.load('penguin_model.pkl'), joblib.load('species_encoder.pkl')

model, le = load_model()

st.title("🐧 Clasificador de Especies")

# Entradas organizadas
with st.sidebar:
    st.header("Características")
    bill_l = st.slider("Longitud Pico (mm)", 30.0, 60.0, 45.0)
    bill_d = st.slider("Profundidad Pico (mm)", 10.0, 25.0, 15.0)
    flip_l = st.slider("Aleta (mm)", 170.0, 240.0, 200.0)
    mass = st.slider("Masa (g)", 2500, 6500, 4000)
    isl = st.selectbox("Isla", ["Biscoe", "Dream", "Torgersen"])
    gender = st.selectbox("Sexo", ["Male", "Female"])

# Lógica de predicción 
if st.button("Predecir Especie"):
    # 1. Preparar datos y Predecir
    input_data = pd.DataFrame({
        'bill_length_mm': [bill_l], 'bill_depth_mm': [bill_d],
        'flipper_length_mm': [flip_l], 'body_mass_g': [mass],
        'island_Dream': [1 if isl == 'Dream' else 0],
        'island_Torgersen': [1 if isl == 'Torgersen' else 0],
        'sex_Male': [1 if gender == 'Male' else 0]
    })
    
    res = model.predict(input_data)
    name = le.inverse_transform(res)[0] # Obtiene 'Adelie', 'Chinstrap' o 'Gentoo'

    # 2. Información y Rutas Locales
    img_path = f"assets/{name.lower()}_penguin.webp"
    
    descripciones = {
        "Adelie": "Se reconoce por el anillo blanco alrededor del ojo y su pico mayoritariamente rojo.",
        "Chinstrap": "Lleva una delgada banda negra bajo la cabeza, como si fuera un casco.",
        "Gentoo": "Es el más grande de los tres y tiene una mancha blanca distintiva sobre cada ojo."
    }

    # 3. Mostrar Resultado 
    st.divider()
    col_img, col_text = st.columns([1, 1.5])

    with col_img:
        # Verificamos si la imagen existe antes de mostrarla para evitar errores
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Imagen {img_path} no encontrada.")

    with col_text:
        st.success(f"{name}")
        st.write(descripciones[name])