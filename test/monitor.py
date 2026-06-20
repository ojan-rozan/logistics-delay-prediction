"""Monitoring: data drift (PSI) + deteksi anomali pengiriman (Isolation Forest)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config as C
from src.utils.logger import get_logger

log = get_logger("monitor")


def population_stability_index(ref: pd.Series, cur: pd.Series, bins=10) -> float:
    edges = np.unique(ref.quantile(np.linspace(0, 1, bins + 1)).values)
    if len(edges) < 2:
        return 0.0
    ref_pct = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), 1e-6, None)
    cur_pct = np.clip(np.histogram(cur, bins=edges)[0] / max(len(cur), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, features, threshold=0.2) -> dict:
    report = {f: {"psi": round(population_stability_index(reference[f], current[f]), 4)}
              for f in features if f in reference and f in current}
    drifted = [f for f, v in report.items() if v["psi"] > threshold]
    log.info("Drift: %s/%s fitur drift", len(drifted), len(report))
    return {"features": report, "drifted": drifted, "needs_retrain": bool(drifted)}


def detect_anomalies(df: pd.DataFrame, contamination=0.03) -> pd.DataFrame:
    """Tandai pengiriman dgn berat/biaya/jarak tak wajar."""
    cols = C.NUM_RAW
    iso = IsolationForest(contamination=contamination, random_state=C.RANDOM_STATE)
    flags = iso.fit_predict(df[cols].fillna(df[cols].median()))
    out = df[flags == -1]
    log.info("Anomali terdeteksi: %s", len(out))
    return out
