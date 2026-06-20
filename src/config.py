"""Konfigurasi terpusat: path, kolom, dan hyperparameter."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"

RAW_FILE = DATA_DIR / "logistics_shipments_dataset.csv"
MODEL_FILE = OUTPUT_DIR / "model_delay_catboost.pkl"
MLP_FILE = OUTPUT_DIR / "model_delay_mlp.pt"
PREPROCESSOR_FILE = OUTPUT_DIR / "preprocessor.pkl"

# MLflow tracking (backend SQLite + artifak di mlruns/)
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
MLFLOW_EXPERIMENT = "logistics_delay"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# === Skema kolom (sesuai logistics_shipments_dataset.csv) ===
COL_SHIP_DATE = "Shipment_Date"
COL_DELIVERY_DATE = "Delivery_Date"   # leakage: hanya diketahui setelah trip
COL_STATUS = "Status"                 # sumber target
POSITIVE_CLASS = "Delayed"
KEEP_STATUS = ["Delivered", "Delayed"]  # pengiriman yang sudah selesai

NUM_RAW = ["Weight_kg", "Cost", "Distance_miles"]
CAT_FEATURES = ["Origin_Warehouse", "Destination", "Carrier"]

# Hyperparameter
CATBOOST_PARAMS = dict(iterations=500, depth=6, learning_rate=0.05, verbose=0)
MLP_PARAMS = dict(max_epochs=100, batch_size=64, lr=1e-3,
                  weight_decay=1e-4, patience=10)
