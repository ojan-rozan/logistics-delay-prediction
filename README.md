# US Logistics — Delivery Delay Prediction

Proyek modular untuk memprediksi keterlambatan pengiriman (klasifikasi biner).
Fokus pada data processing (missing values, outlier, encoding, scaling),
penanganan class imbalance, CatBoost + PyTorch MLP, serving via FastAPI.

## Struktur

```
.
├── data/                 # taruh logistics_shipments_dataset.csv di sini
├── notebook/             # EDA & eksperimen
├── output/               # model & metrics hasil training
├── logs/                 # log aplikasi
├── mlruns/               # tracking eksperimen (opsional)
├── src/
│   ├── config.py         # path, nama kolom, hyperparameter
│   ├── data/
│   │   ├── data_loader.py        # load CSV
│   │   └── data_processing.py    # target, feature eng, preprocessor (anti-leakage)
│   ├── models/
│   │   ├── model.py              # pipeline CatBoost + arsitektur MLP
│   │   └── trainer.py            # training, CV, evaluasi, save
│   └── utils/
│       ├── logger.py
│       └── metrics.py            # accuracy, precision, recall, F1, ROC-AUC
├── test/
│   ├── train_model.py    # entrypoint training
│   ├── app.py            # FastAPI serving (/predict)
│   ├── monitor.py        # drift (PSI) + deteksi anomali (Isolation Forest)
│   └── retrain_trigger.py# trigger retraining otomatis
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Cara pakai

```bash
pip install -r requirements.txt

# 1. Letakkan dataset di data/

# 2. Training
python -m test.train_model

# 3. Serving API
uvicorn test.app:app --reload
# buka http://localhost:8000/docs

# 4. Docker
docker compose up --build
```

## Contoh request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Shipment_Date":"2023-07-15","Origin_Warehouse":"Warehouse_LA","Destination":"Chicago","Carrier":"FedEx","Weight_kg":12.5,"Cost":85,"Distance_miles":750}'
```

## Catatan teknis

- **Target**: `Status` difilter ke `Delivered`/`Delayed`; `target = 1` jika `Delayed`. `In Transit`/`Lost` dibuang (belum ada hasil).
- **Anti-leakage**: `Delivery_Date` & `Status` tidak dipakai sebagai fitur.
- **Processing**: imputasi missing values, `RobustScaler` (tahan outlier), one-hot encoding kategori.
- **Class imbalance**: `scale_pos_weight` (CatBoost) & `pos_weight` (MLP), evaluasi dengan F1 & ROC-AUC.
- **Model produksi**: CatBoost (pipeline lengkap, joblib). **Deep learning**: PyTorch MLP.
- Verifikasi nama kolom dgn `print(df.columns.tolist())` — dataset disebut punya 11 kolom; tambahkan ke `config.py` jika ada kolom relevan lain.

## Streamlit App (UI)

```bash
python -m test.train_model      # pastikan model sudah ada di output/
streamlit run streamlit_app.py  # buka http://localhost:8501
```

Fitur: prediksi satuan (form input) + prediksi batch (upload CSV, download hasil).

## MLflow Tracking

Training otomatis mencatat parameter, metrik, dan model ke MLflow (backend SQLite, artifak di `mlruns/`).

```bash
python -m test.train_model                              # jalankan training (otomatis ter-log)
mlflow ui --backend-store-uri sqlite:///mlflow.db       # buka dashboard
# lalu buka http://localhost:5000
```

Setiap run tercatat sebagai `catboost` & `mlp` lengkap dengan hyperparameter, F1/ROC-AUC, kurva loss/AUC per epoch, dan grafik training.

