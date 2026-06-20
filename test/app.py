"""API serving (FastAPI). Input data pengiriman -> probabilitas & label keterlambatan.

Jalankan: uvicorn test.app:app --host 0.0.0.0 --port 8000
"""
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config as C
from src.data import data_processing as dp
from src.utils.logger import get_logger

log = get_logger("api")
app = FastAPI(title="Logistics Delay API", version="1.0")


class ShipmentInput(BaseModel):
    Shipment_Date: str = Field(..., examples=["2023-07-15"])
    Origin_Warehouse: str = "Warehouse_LA"
    Destination: str = "Chicago"
    Carrier: str = "FedEx"
    Weight_kg: float = 12.5
    Cost: float = 85.0
    Distance_miles: float = 750.0


@lru_cache
def get_model():
    if not C.MODEL_FILE.exists():
        raise FileNotFoundError("Model belum dilatih. Jalankan: python -m test.train_model")
    return joblib.load(C.MODEL_FILE)


def _to_frame(inp: ShipmentInput) -> pd.DataFrame:
    row = {
        C.COL_SHIP_DATE: inp.Shipment_Date,
        "Origin_Warehouse": inp.Origin_Warehouse,
        "Destination": inp.Destination,
        "Carrier": inp.Carrier,
        "Weight_kg": inp.Weight_kg,
        "Cost": inp.Cost,
        "Distance_miles": inp.Distance_miles,
    }
    return dp.add_features(pd.DataFrame([row]))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": C.MODEL_FILE.exists()}


@app.post("/predict")
def predict(inp: ShipmentInput):
    try:
        model = get_model()
        X = _to_frame(inp)
        proba = float(model.predict_proba(X)[0, 1])
        return {"delay_probability": round(proba, 4),
                "prediction": "Delayed" if proba >= 0.5 else "On-time"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=400, detail=str(e))
