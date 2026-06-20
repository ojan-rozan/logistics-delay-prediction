"""Target, cleaning, feature engineering, dan preprocessor (anti-leakage)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src import config as C
from src.utils.logger import get_logger

log = get_logger(__name__)

# Kolom yang hanya diketahui SETELAH pengiriman -> tidak boleh jadi fitur.
LEAKAGE_COLS = [C.COL_DELIVERY_DATE, C.COL_STATUS]

DATE_FEATURES = ["ship_month", "ship_dayofweek", "ship_is_weekend",
                 "ship_quarter", "ship_is_winter"]


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df[df[C.COL_STATUS].isin(C.KEEP_STATUS)].copy()
    df["target"] = (df[C.COL_STATUS] == C.POSITIVE_CLASS).astype(int)
    log.info("Target dibuat. Distribusi: %s", df["target"].value_counts().to_dict())
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering tanggal. Reusable utk training & inference."""
    df = df.copy()
    df[C.COL_SHIP_DATE] = pd.to_datetime(df[C.COL_SHIP_DATE], errors="coerce")
    d = df[C.COL_SHIP_DATE].dt
    df["ship_month"] = d.month
    df["ship_dayofweek"] = d.dayofweek
    df["ship_is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["ship_quarter"] = d.quarter
    df["ship_is_winter"] = d.month.isin([12, 1, 2]).astype(int)
    return df


def get_feature_lists():
    num = C.NUM_RAW + DATE_FEATURES
    cat = C.CAT_FEATURES
    return num, cat


def build_preprocessor(num, cat) -> ColumnTransformer:
    # RobustScaler dipilih karena tahan outlier (dataset punya ~3% outlier).
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", RobustScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def prepare(df: pd.DataFrame):
    df = create_target(df)
    df = add_features(df)
    num, cat = get_feature_lists()
    for c in cat:
        df[c] = df[c].astype(str)
    X = df[num + cat].copy()
    y = df["target"].copy()
    return X, y, num, cat


def split(X, y):
    return train_test_split(X, y, test_size=C.TEST_SIZE,
                            stratify=y, random_state=C.RANDOM_STATE)
