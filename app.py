from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

current_model = None
current_model_version = None

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)


class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class UpdateModelRequest(BaseModel):
    version: str


def load_model_from_mlflow(version: str = "latest"):
    global current_model, current_model_version

    try:
        model_uri = f"models:/iris_logistic_regression/{version}"
        model = mlflow.sklearn.load_model(model_uri)

        current_model = model
        current_model_version = version

        return True
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    print("Démarrage du service")
    load_model_from_mlflow("latest")


@app.get("/")
async def root():
    print("Hello World")


@app.post("/predict")
async def predict(request: PredictionRequest):
    if current_model is None:
        return {"error": "Modèle non chargé"}

    try:
        features = np.array(
            [
                [
                    request.sepal_length,
                    request.sepal_width,
                    request.petal_length,
                    request.petal_width,
                ]
            ]
        )

        prediction = current_model.predict(features)[0]

        return {"prediction": int(prediction), "model_version": current_model_version}
    except Exception as e:
        return {"error": f"Erreur de prédiction: {str(e)}"}


@app.post("/update-model")
async def update_model(request: UpdateModelRequest):
    success = load_model_from_mlflow(request.version)

    if success:
        return {"message": "Modèle mis à jour avec succès", "version": request.version}
    else:
        return {
            "error": "Échec de la mise à jour du modèle",
            "version": request.version,
        }
