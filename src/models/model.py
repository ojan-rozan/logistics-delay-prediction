"""Definisi model: pipeline CatBoost (klasifikasi) dan PyTorch MLP."""
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
import torch.nn as nn

from src import config as C


def build_catboost_pipeline(preprocessor, scale_pos_weight=1.0) -> Pipeline:
    """Preprocessor + CatBoost. scale_pos_weight menangani imbalance."""
    model = CatBoostClassifier(scale_pos_weight=scale_pos_weight,
                               random_seed=C.RANDOM_STATE, **C.CATBOOST_PARAMS)
    return Pipeline([("prep", preprocessor), ("model", model)])


class MLPClassifier(nn.Module):
    """MLP untuk klasifikasi biner (output logit, dipakai dgn BCEWithLogitsLoss)."""

    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)
