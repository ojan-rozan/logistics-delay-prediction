"""Load data mentah dari CSV."""
import pandas as pd
from src import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_FILE
    df = pd.read_csv(path)
    log.info("Loaded %s rows from %s", len(df), path)
    return df
