import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# docker build -t pm10-api .
# docker run -p 8000:8000 pm10-api
# http://localhost:8000/docs
# http://127.1.0.1:8000/docs

# KONSTANTE
WINDOW_SIZE = 72
SAVED_DIR = "saved"

# NALOŽI Modele in scaler
model = tf.keras.models.load_model(f"{SAVED_DIR}/final_lstm_model.keras")
scaler_X = joblib.load(f"{SAVED_DIR}/scaler_X.pkl")
selected_features = joblib.load(f"{SAVED_DIR}/selected_features.pkl")

app = FastAPI(title="PM10 Forecast API", version="1.0")


# INPUT SCHEMA
class PredictRequest(BaseModel):
    # Seznam 72 ur, vsaka ura je dict feature -> vrednost
    records: List[Dict[str, float]]


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Preveri dolžino okna
        if len(req.records) != WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {WINDOW_SIZE} records (hours), but got {len(req.records)}"
            )

        # Pretvori v DataFrame
        df = pd.DataFrame(req.records)

        # Poravnava stolpcev (če kak manjka -> error; če je extra -> ignoriramo)
        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {missing}"
            )

        df = df[selected_features]  # pravilen vrstni red

        # Scale + reshape za model: (1, 72, 11)
        X_scaled = scaler_X.transform(df.values)
        X_seq = X_scaled.reshape(1, WINDOW_SIZE, len(selected_features))

        # Predict (v log prostoru)
        pred_log = model.predict(X_seq, verbose=0).reshape(-1)[0]

        # Nazaj v realni PM10
        pred_pm10 = float(np.expm1(pred_log))

        return {"prediction": round(pred_pm10, 2)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
