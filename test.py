import pandas as pd
import joblib

KNN_PATH = "python-scripts/production_model/modelo_knn.pkl"
REGRESION_PATH = "python-scripts/production_model/modelo_regresion.pkl"

def cargar_modelos():
    try:
        pipeline_knn = joblib.load(KNN_PATH)
        pipeline_regresion = joblib.load(REGRESION_PATH)
        print("Modelos cargados correctamente.")
        return pipeline_knn, pipeline_regresion
    except:
        print("Error: primero debes ejecutar train.py")
        exit()


def predecir(pipeline_knn, pipeline_regresion, frecuencia_cardiaca, potencia, cadencia, tiempo, temperatura, pendiente, velocidad):
    datos = pd.DataFrame(
        [[frecuencia_cardiaca, potencia, cadencia, tiempo, temperatura, pendiente, velocidad]],
        columns=["frecuencia_cardiaca", "potencia", "cadencia", "tiempo", "temperatura", "pendiente", "velocidad"]
    )
    pred_knn = pipeline_knn.predict(datos)
    pred_regresion = pipeline_regresion.predict(datos)
    return round(pred_knn[0], 2), round(pred_regresion[0], 2)
