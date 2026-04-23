import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


KNN_PATH       = "modelo_knn.pkl"
REGRESION_PATH = "modelo_regresion.pkl"

def entrenar(porcentaje_test=0.2, k=5):

    data = pd.read_csv("data/dataset_ciclismo_fatiga.csv")

    X = data[['frecuencia_cardiaca', 'potencia', 'cadencia', 'tiempo', 'temperatura', 'pendiente', 'velocidad']]
    y = data["fatiga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=porcentaje_test, random_state=42
    )

    pipeline_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsRegressor(n_neighbors=k))
    ])

    pipeline_regresion = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LinearRegression())
    ])

    pipeline_knn.fit(X_train, y_train)
    pipeline_regresion.fit(X_train, y_train)

    y_pred_knn = pipeline_knn.predict(X_test)
    y_pred_regresion = pipeline_regresion.predict(X_test)

    metricas_knn = {
        "MSE": round(mean_squared_error(y_test, y_pred_knn), 2),
        "R²":  round(r2_score(y_test, y_pred_knn), 4),
    }

    joblib.dump(metricas_knn, "metricas_knn.pkl")
    
    metricas_regresion = {
         "MSE": round(mean_squared_error(y_test, y_pred_regresion), 2),
         "R²": round(r2_score(y_test, y_pred_regresion), 4)
    }

    joblib.dump(metricas_regresion, "metricas_regresion.pkl")

    joblib.dump(pipeline_knn, KNN_PATH)
    joblib.dump(pipeline_regresion, REGRESION_PATH)
    
    return pipeline_knn, pipeline_regresion,metricas_knn, metricas_regresion


if __name__ == "__main__":
    entrenar()
