from typing import Dict, Tuple, List
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def active_cr_card_func(d: pd.DataFrame) -> pd.DataFrame:
    """
    Create new feature ActiveCrCard using HasCrCard and IsActiveMember.

    Parameters
    ----------
    d : pd.DataFrame
        Input dataframe with columns 'HasCrCard' and 'IsActiveMember'.

    Returns
    -------
    pd.DataFrame
        Dataframe with single column 'ActiveCrCard'.
    """
    d["ActiveCrCard"] = d["HasCrCard"] + (4 * d["IsActiveMember"]) + 1
    return d[["ActiveCrCard"]]

def tenure_by_age_func(d: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature TenureAgeR as Tenure divided by log(Age).

    Parameters
    ----------
    d : pd.DataFrame
        Input dataframe with columns 'Age' and 'Tenure'.

    Returns
    -------
    pd.DataFrame
        Dataframe with single column 'TenureAgeR'.
    """
    d["Age"] = np.log(d["Age"])
    d["TenureAgeR"] = d["Tenure"] / d["Age"]
    return d[["TenureAgeR"]]

def log_age_func(d: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to Age column.

    Parameters
    ----------
    d : pd.DataFrame
        Input dataframe with column 'Age'.

    Returns
    -------
    pd.DataFrame
        Dataframe with transformed 'Age' column.
    """
    d["Age"] = np.log(d["Age"])
    return d[["Age"]]

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering with support for feature names.

    Parameters
    ----------
    func : callable
        Function to transform dataframe.
    columns : list
        List of column names expected by the function.
    feature_name : str
        Output feature name after transformation.
    """
    def __init__(self, func, columns: List[str], feature_name: str):
        self.func = func
        self.columns = columns
        self.feature_name = feature_name
        self._is_fitted = False

    def fit(self, X, y=None):
        """Fit method (no-op, just set flag)."""
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform method applying the custom function."""
        df = pd.DataFrame(X, columns=self.columns)
        result = self.func(df)
        return result.values

    def get_feature_names_out(self, input_features=None):
        """Return output feature name."""
        if not self._is_fitted:
            raise AttributeError("This transformer is not fitted yet. Call 'fit' first.")
        return [self.feature_name]

def create_custom_feature_pipeline(func, columns: List[str], feature_name: str) -> Pipeline:
    """
    Create a pipeline for a custom feature.

    Parameters
    ----------
    func : callable
        Function to create feature.
    columns : list
        Columns used by the function.
    feature_name : str
        Name of the resulting feature.

    Returns
    -------
    Pipeline
        Pipeline object for custom feature engineering.
    """
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median"))
    ])
    transformer = CustomFunctionTransformer(func, columns, feature_name)
    return make_pipeline(num_pipeline, transformer)

def get_custom_pipelines() -> Dict[str, Tuple[Pipeline, List[str]]]:
    """
    Create all custom feature pipelines.

    Returns
    -------
    dict
        Dictionary mapping feature name to tuple of pipeline and columns.
    """
    return {
        "ActiveCrCard": (create_custom_feature_pipeline(active_cr_card_func, ["IsActiveMember", "HasCrCard"], "ActiveCrCard"), ["IsActiveMember", "HasCrCard"]),
        "TenureAgeR": (create_custom_feature_pipeline(tenure_by_age_func, ["Age", "Tenure"], "TenureAgeR"), ["Age", "Tenure"]),
        "Age": (create_custom_feature_pipeline(log_age_func, ["Age"], "Age"), ["Age"])
    }
