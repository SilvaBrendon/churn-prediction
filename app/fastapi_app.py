from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Inicializar app FastAPI
app = FastAPI(title="API de Churn", description="Previsão de cancelamento de clientes", version="1.0")

# Caminho dos arquivos salvos
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocessor.pkl'))

# Classe para definir os campos esperados na requisição
class Cliente(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"mensagem": "API de previsão de churn está ativa!"}

@app.post("/predict")
def prever_churn(cliente: Cliente):
    # Transformar dados em DataFrame
    df = pd.DataFrame([cliente.dict()])

    # Aplicar pré-processamento
    X = preprocessor.transform(df)

    # Fazer a previsão
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    # Responder
    return {
        "churn": bool(pred),
        "probabilidade": round(prob, 4)
    }
