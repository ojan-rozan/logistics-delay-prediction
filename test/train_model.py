"""Entrypoint training: load -> prepare -> train (CatBoost + MLP) -> evaluate -> save.

Jalankan dari root: python -m test.train_model
"""
import json

import mlflow


from src import config as C
from src.data.data_loader import load_raw
from src.data import data_processing as dp
from src.models import trainer
from src.utils.logger import get_logger
from src.utils.tracking import init_mlflow

log = get_logger("train_model")


def _save_training_curve(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, history["train_loss"], "tab:red", marker="o", ms=3, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_auc"], "tab:blue", marker="s", ms=3, label="Val ROC-AUC")
    ax2.axvline(history["best_epoch"], color="green", ls="--", alpha=0.7,
                label=f"Best epoch ({history['best_epoch']})")
    ax2.set_ylabel("Val ROC-AUC", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="lower right")
    plt.title("Training Curve + Early Stopping"); fig.tight_layout()
    out = C.OUTPUT_DIR / "training_curve.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    log.info("Saved training curve -> %s", out)


def main():
    init_mlflow()

    df = load_raw()
    X, y, num, cat = dp.prepare(df)
    X_train, X_test, y_train, y_test = dp.split(X, y)

    # --- CatBoost (model produksi) ---
    with mlflow.start_run(run_name="catboost"):
        cb = trainer.train_catboost(X_train, y_train, num, cat)
        cv_f1 = trainer.cross_validate(cb, X_train, y_train)
        cb_metrics = trainer.evaluate(cb, X_test, y_test)
        log.info("CatBoost: %s", cb_metrics)
        trainer.save_catboost(cb)

        mlflow.log_params(C.CATBOOST_PARAMS)
        mlflow.log_metric("cv_f1", cv_f1)
        mlflow.log_metrics({f"test_{k}": v for k, v in cb_metrics.items()})
        mlflow.log_artifact(str(C.MODEL_FILE))

    # --- MLP (deep learning) dengan early stopping ---
    with mlflow.start_run(run_name="mlp"):
        preprocessor = dp.build_preprocessor(num, cat)
        mlp, mlp_metrics, history = trainer.train_mlp(
            preprocessor, X_train, X_test, y_train, y_test)
        log.info("MLP: %s (best epoch %d)", mlp_metrics, history["best_epoch"])
        trainer.save_mlp(mlp, preprocessor)
        _save_training_curve(history)

        mlflow.log_params(C.MLP_PARAMS)
        mlflow.log_param("best_epoch", history["best_epoch"])
        mlflow.log_metrics({f"test_{k}": v for k, v in mlp_metrics.items()})
        # kurva loss/AUC per epoch
        for i, (l, a) in enumerate(zip(history["train_loss"], history["val_auc"]), 1):
            mlflow.log_metrics({"train_loss": l, "val_auc": a}, step=i)
        mlflow.log_artifact(str(C.OUTPUT_DIR / "training_curve.png"))

    results = {"catboost": cb_metrics, "mlp": mlp_metrics}
    (C.OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info("Done. Metrics -> output/metrics.json | MLflow runs -> mlruns/")


if __name__ == "__main__":
    main()
