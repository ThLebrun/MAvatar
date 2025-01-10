"""FAMD projection module."""
import sys
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from saiph.models import Model
from saiph.reduction import DUMMIES_SEPARATOR
from saiph.reduction.famd import fit as fit_famd
from saiph.reduction.famd import transform as transform_famd


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float64]] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Model:
    """Fit a FAMD model on sparse data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )
    return fit_famd(df, nf, col_weights, center=center_sparse, seed=random_gen)


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float64]] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a FAMD model on data and return transformed data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        coord: The transformed data.
        model: The model for transforming new data.
    """
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )

    model = fit(df, nf, col_weights, seed=random_gen)
    coord = transform(df, model)
    return coord, model


def center_sparse(
    df: pd.DataFrame, quanti: List[str], quali: List[str]
) -> Tuple[
    scipy.sparse.spmatrix,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[Any],
    NDArray[Any],
]:
    """Center data, scale it, compute modalities and proportions of each categorical.

    Used as internal function during fit.

    **NB**: saiph.reduction.famd.scaler is better suited when a Model is already fitted.

    Parameters:
        df: DataFrame to center.
        quanti: Indices of continuous variables.
        quali: Indices of categorical variables.

    Returns:
        df_scale: The scaled DataFrame.
        mean: Mean of the input dataframe.
        std: Standard deviation of the input dataframe.
        prop: Proportion of each categorical.
        _modalities: Modalities for the MCA.
    """
    # Scale the continuous data
    df_quanti = df[quanti]
    mean = np.mean(df_quanti, axis=0)
    df_quanti -= mean
    std = np.std(df_quanti, axis=0)
    std_without_zero = np.where(std <= sys.float_info.min, 1, std)
    df_quanti /= std_without_zero
    df_quanti = csr_matrix(df_quanti)

    # scale the categorical data
    df_quali = pd.get_dummies(
        df[quali].astype("category"),
        prefix_sep=DUMMIES_SEPARATOR,
        dtype=np.uint8,
    )
    _modalities = df_quali.columns
    df_quali = csr_matrix(df_quali)

    prop = np.mean(df_quali, axis=0).tolist()[0]
    df_quali /= np.sqrt(prop)
    df_scale = scipy.sparse.hstack([df_quanti, df_quali], format="csr")
    return df_scale, mean, std, prop, _modalities


def scaler_sparse(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Scale data using mean, std, modalities and proportions of each categorical from model.

    Parameters:
        model: Model computed by fit.
        df: DataFrame to scale.

    Returns:
        df_scaled: The scaled DataFrame.
    """
    model.prop = cast(pd.Series, model.prop)
    df_quanti = df[model.original_continuous]
    df_quanti = (df_quanti - model.mean) / model.std
    df_quanti = scipy.sparse.csr_matrix(df_quanti)

    # scale
    df_quali = pd.get_dummies(
        df[model.original_categorical].astype("category"), prefix_sep=DUMMIES_SEPARATOR
    )
    if model._modalities is not None:
        for mod in model._modalities:
            if mod not in df_quali:
                df_quali[mod] = 0
    df_quali = df_quali[model._modalities]
    df_quali = csr_matrix(df_quali)
    df_quali /= np.sqrt(model.prop)

    df_scaled = scipy.sparse.hstack([df_quanti, df_quali], format="csr")
    return df_scaled


def transform(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space.
    """
    return transform_famd(df, model, scaler=scaler_sparse)
