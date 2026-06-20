"""Setup MLflow tracking (backend SQLite, artifak disimpan di mlruns/)."""
import mlflow

from src import config as C


def init_mlflow():
    C.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(C.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(C.MLFLOW_EXPERIMENT,
                                 artifact_location=C.MLRUNS_DIR.as_uri())
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
