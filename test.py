import pandas as pd
import joblib
import os

KNN_PATH = "modelo_knn.pkl"
REGRESION_PATH = "modelo_regresion.pkl"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cargar_modelos():
    try:
        pipeline_knn = joblib.load(os.path.join(BASE_DIR, KNN_PATH))
        pipeline_regresion = joblib.load(os.path.join(BASE_DIR, REGRESION_PATH))
        print("Modelos cargados correctamente.")
        return pipeline_knn, pipeline_regresion
    except Exception as e:
        raise Exception(f"No se pudieron cargar los modelos: {e}")


def predecir(pipeline_knn, pipeline_regresion, frecuencia_cardiaca, potencia, cadencia, tiempo, temperatura, pendiente, velocidad):
    datos = pd.DataFrame(
        [[frecuencia_cardiaca, potencia, cadencia, tiempo, temperatura, pendiente, velocidad]],
        columns=["frecuencia_cardiaca", "potencia", "cadencia", "tiempo", "temperatura", "pendiente", "velocidad"]
    )
    pred_knn = pipeline_knn.predict(datos)
    pred_regresion = pipeline_regresion.predict(datos)
    return round(pred_knn[0], 2), round(pred_regresion[0], 2)
