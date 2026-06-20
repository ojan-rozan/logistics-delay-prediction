"""Trigger retraining bila drift terdeteksi atau F1 turun di bawah threshold.

Jalankan: python -m test.retrain_trigger
"""
from src.data.data_loader import load_raw
from src.data import data_processing as dp
from src.models import trainer
from test.monitor import detect_drift
from src.utils.logger import get_logger

log = get_logger("retrain")

F1_THRESHOLD = 0.55  # retrain jika F1 test di bawah ini


def main():
    df = load_raw()
    X, y, num, cat = dp.prepare(df)
    X_train, X_test, y_train, y_test = dp.split(X, y)

    cb = trainer.train_catboost(X_train, y_train, num, cat)
    metrics = trainer.evaluate(cb, X_test, y_test)
    drift = detect_drift(X_train, X_test, num)["needs_retrain"]

    if drift or metrics["f1"] < F1_THRESHOLD:
        log.info("Kondisi retrain terpenuhi (drift=%s, F1=%.3f). Simpan model baru.",
                 drift, metrics["f1"])
        trainer.save_catboost(cb)
    else:
        log.info("Model sehat (F1=%.3f). Tidak retrain.", metrics["f1"])


if __name__ == "__main__":
    main()
