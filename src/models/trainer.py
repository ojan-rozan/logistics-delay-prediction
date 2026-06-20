"""Training, validasi silang, evaluasi, dan penyimpanan model."""
import copy

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data import data_processing as dp
from src.models.model import build_catboost_pipeline, MLPClassifier
from src.utils.metrics import classification_metrics
from src.utils.logger import get_logger

log = get_logger(__name__)


def _pos_weight(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def train_catboost(X_train, y_train, num, cat):
    spw = _pos_weight(y_train)
    pipe = build_catboost_pipeline(dp.build_preprocessor(num, cat), scale_pos_weight=spw)
    pipe.fit(X_train, y_train)
    return pipe


def cross_validate(pipe, X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    log.info("CV F1: %.3f +/- %.3f", scores.mean(), scores.std())
    return scores.mean()


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    return classification_metrics(y_test, model.predict(X_test), proba)


def train_mlp(preprocessor, X_train, X_test, y_train, y_test):
    """Latih MLP dengan early stopping (pantau Val ROC-AUC) + weight decay.

    Mengembalikan: (model_terbaik, metrics, history) di mana history =
    {'train_loss': [...], 'val_auc': [...], 'best_epoch': int}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = preprocessor.fit_transform(X_train).astype("float32")
    Xte = preprocessor.transform(X_test).astype("float32")
    ytr = y_train.values.astype("float32").reshape(-1, 1)

    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=C.MLP_PARAMS["batch_size"], shuffle=True)
    model = MLPClassifier(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.MLP_PARAMS["lr"],
                           weight_decay=C.MLP_PARAMS["weight_decay"])
    pos_weight = torch.tensor([_pos_weight(y_train)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loss, val_auc = [], []
    best_auc, best_state, best_ep, wait = -1.0, None, 0, 0
    Xte_t = torch.tensor(Xte).to(device)

    for ep in range(1, C.MLP_PARAMS["max_epochs"] + 1):
        model.train(); total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item() * len(xb)
        train_loss.append(total / len(dl.dataset))

        model.eval()
        with torch.no_grad():
            proba = 1 / (1 + np.exp(-model(Xte_t).cpu().numpy().ravel()))
        auc = roc_auc_score(y_test, proba)
        val_auc.append(auc)

        if auc > best_auc + 1e-4:
            best_auc, best_state, best_ep, wait = auc, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= C.MLP_PARAMS["patience"]:
                log.info("Early stop di epoch %d | best Val AUC %.3f (epoch %d)",
                         ep, best_auc, best_ep)
                break

    model.load_state_dict(best_state)  # pulihkan bobot terbaik
    model.eval()
    with torch.no_grad():
        proba = 1 / (1 + np.exp(-model(Xte_t).cpu().numpy().ravel()))
    pred = (proba >= 0.5).astype(int)
    history = {"train_loss": train_loss, "val_auc": val_auc, "best_epoch": best_ep}
    return model, classification_metrics(y_test, pred, proba), history


def save_catboost(pipe):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, C.MODEL_FILE)
    log.info("Saved CatBoost pipeline -> %s", C.MODEL_FILE)


def save_mlp(model, preprocessor):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), C.MLP_FILE)
    joblib.dump(preprocessor, C.PREPROCESSOR_FILE)
    log.info("Saved MLP -> %s", C.MLP_FILE)
