import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

def create_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_enc_cols: List[str],
    custom_pipelines: Dict[str, Tuple[Pipeline, List[str]]]
) -> ColumnTransformer:
    """
    Create a full preprocessing pipeline with numeric, categorical, target encoding and custom pipelines.

    Parameters
    ----------
    numeric_cols : list
        List of numeric feature columns.
    categorical_cols : list
        List of categorical feature columns.
    target_enc_cols : list
        Columns for target encoding.
    custom_pipelines : dict
        Dictionary of custom pipelines with columns.

    Returns
    -------
    ColumnTransformer
        Configured column transformer.
    """
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median"))
    ])

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))
    ])

    target_enc_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", TargetEncoder())
    ])

    transformers = [
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
        ("target", target_enc_pipeline, target_enc_cols)
    ]

    for name, (pipeline, columns) in custom_pipelines.items():
        transformers.append((name, pipeline, columns))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

def split_data(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Split dataframe into train and validation sets and define input columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Target column name.
    drop_cols : list, optional
        Columns to drop.

    Returns
    -------
    tuple
        X_train, y_train, X_val, y_val, list of input columns.
    """
    if drop_cols is None:
        drop_cols = ["id", "CustomerId", target_col]

    input_cols = [col for col in df.columns if col not in drop_cols]
    X = df[input_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, y_train, X_val, y_val, input_cols

def transform_data(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Fit preprocessor on train set and transform train and validation sets.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    y_train : pd.Series
        Training targets.

    Returns
    -------
    tuple
        Transformed X_train, X_val, and list of feature names.
    """
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    feature_names = list(preprocessor.get_feature_names_out())

    # Correct names for TargetEncoder columns
    target_cols = preprocessor.transformers_[2][2].copy()
    for i, name in enumerate(feature_names):
        if isinstance(name, int) or name in ["0", "1", "2"]:
            feature_names[i] = target_cols.pop(0)

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_processed, columns=feature_names, index=X_val.index)

    return X_train_df, X_val_df, feature_names

def preprocess_data(
    raw_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    target_col: str = "Exited",
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Legacy main function to preprocess data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Input dataframe.
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
    target_col : str, optional
        Target column name, default "Exited".
    drop_cols : list, optional
        Columns to drop.

    Returns
    -------
    tuple
        Transformed X_train, y_train, X_val, y_val, and feature names.
    """
    X_train, y_train, X_val, y_val, input_cols = split_data(raw_df, target_col, drop_cols)
    X_train_df, X_val_df, feature_names = transform_data(preprocessor, X_train, X_val, y_train)
    return X_train_df, y_train, X_val_df, y_val, feature_names

def preprocess_new_data(
    raw_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    drop_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Preprocess new unseen data using an already fitted preprocessor.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw dataframe to preprocess.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    drop_cols : list, optional
        Columns to drop.

    Returns
    -------
    pd.DataFrame
        Transformed dataframe ready for prediction.
    """
    if drop_cols is None:
        drop_cols = ["id", "CustomerId", "Surname", "Exited"]

    input_cols = [col for col in raw_df.columns if col not in drop_cols]
    X = raw_df[input_cols]

    X_processed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    # Correct TargetEncoder column names
    target_cols = preprocessor.transformers_[2][2].copy()
    for i, name in enumerate(feature_names):
        if isinstance(name, int) or name in ["0", "1", "2"]:
            feature_names[i] = target_cols.pop(0)

    X_df = pd.DataFrame(X_processed, columns=feature_names, index=raw_df.index)

    return X_df
