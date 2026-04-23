import streamlit as st
from train import entrenar
from test import cargar_modelos, predecir
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(page_title="Fatiga en Ciclismo · KNN vs Regresión", layout="centered")
 
st.title("Predicción de Fatiga en Ciclismo")

@st.cache_resource
def cargar():
    st.write("Entrando a cargar_modelos()")
    modelos = cargar_modelos()
    st.write("Modelos cargados")
    return modelos

try:
    st.write("Antes de cargar modelos")
    pipeline_knn, pipeline_regresion = cargar()
    st.write("Después de cargar modelos")
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()
 
# ── Métricas ──────────────────────────────────────────────────────────────────
st.subheader("Métricas de Evaluación")
 
col1, col2 = st.columns(2)

try:
    metricas_knn = joblib.load(os.path.join(BASE_DIR, "metricas_knn.pkl"))
    metricas_regresion = joblib.load(os.path.join(BASE_DIR, "metricas_regresion.pkl"))
except Exception as e:
    st.error(f"Error cargando métricas: {e}")
    st.stop()

with col1:
    st.markdown("**KNN**")
    st.metric("MSE", metricas_knn["MSE"])
    st.metric("R²",  metricas_knn["R²"])
with col2:
    st.markdown("**Regresión Lineal**")
    st.metric("MSE", metricas_regresion["MSE"])
    st.metric("R²",  metricas_regresion["R²"])
 
st.divider()
 
# ── Formulario de predicción (test.py) ────────────────────────────────────────
st.subheader("Predicción de Fatiga")
 
col_a, col_b = st.columns(2)
with col_a:
    frecuencia_cardiaca = st.number_input("Frecuencia cardíaca (bpm)", min_value=0.0, step=1.0)
    potencia            = st.number_input("Potencia (W)",               min_value=0.0, step=1.0)
    cadencia            = st.number_input("Cadencia (rpm)",             min_value=0.0, step=1.0)
    tiempo              = st.number_input("Tiempo (min)",               min_value=0.0, step=0.1)
with col_b:
    temperatura = st.number_input("Temperatura (°C)", min_value=0.0, step=0.1)
    pendiente   = st.number_input("Pendiente (%)",                   step=0.1)
    velocidad   = st.number_input("Velocidad (km/h)", min_value=0.0, step=0.1)
 
if st.button("Predecir Fatiga", type="primary"):
    # Delega la predicción a test.py
    pred_knn, pred_reg = predecir(
        pipeline_knn, pipeline_regresion,
        frecuencia_cardiaca, potencia, cadencia,
        tiempo, temperatura, pendiente, velocidad,
    )
    res1, res2 = st.columns(2)
    with res1:
        st.success(f"**KNN**: {pred_knn}")
    with res2:
        st.info(f"**Regresión Lineal**: {pred_reg}")
 
st.divider()
