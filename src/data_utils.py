import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import Bunch
import joblib

from .config import (
    ADULT_TEST_URL,
    ADULT_URL,
    DATA_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    SENSITIVE_COL,
    TARGET_COL,
    TEST_SIZE,
)

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_adult_raw() -> pd.DataFrame:
    """Load Adult dataset from UCI and return a single cleaned DataFrame."""
    train_df = pd.read_csv(
        ADULT_URL,
        names=ADULT_COLUMNS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
    )
    test_df = pd.read_csv(
        ADULT_TEST_URL,
        names=ADULT_COLUMNS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    # Normalize labels in test split which include a trailing period
    test_df[TARGET_COL] = test_df[TARGET_COL].str.replace(".", "", regex=False)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.dropna(axis=0).reset_index(drop=True)
    return df


def preprocess_adult(df: pd.DataFrame) -> Bunch:
    """Encode, scale, and split Adult data into train/test sets."""
    df = df.copy()
    df[TARGET_COL] = (df[TARGET_COL] == ">50K").astype(int)

    sensitive = df[SENSITIVE_COL].astype(str)
    s = (sensitive == "Male").astype(int).to_numpy()

    X = df.drop(columns=[TARGET_COL, SENSITIVE_COL])
    y = df[TARGET_COL].to_numpy()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor, categorical_cols, numeric_cols)

    return Bunch(
        X_train=X_train_enc,
        X_test=X_test_enc,
        y_train=y_train,
        y_test=y_test,
        s_train=s_train,
        s_test=s_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
    )


def get_feature_names(preprocessor: ColumnTransformer, cat_cols, num_cols):
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    num_features = np.array(num_cols, dtype=object)
    return np.concatenate([cat_features, num_features])


def prepare_adult_dataset(force: bool = False) -> Bunch:
    """Load and preprocess Adult dataset, with cached artifacts."""
    ensure_dirs()
    cache_path = PROCESSED_DIR / "adult_processed.joblib"
    if cache_path.exists() and not force:
        return joblib.load(cache_path)

    df = load_adult_raw()
    bundle = preprocess_adult(df)
    joblib.dump(bundle, cache_path)
    return bundle


def save_dataset_stats(stats: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
