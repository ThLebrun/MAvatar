"""PCA projection module."""
import sys
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction.utils.common import (
    get_explained_variance,
    get_projected_column_names,
    get_uniform_row_weights,
)
from saiph.reduction.utils.svd import get_svd


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float64]] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Model:
    """Fit a PCA model on data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    nf = nf or min(df.shape)
    _col_weights = col_weights if col_weights is not None else np.ones(df.shape[1])
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )

    # Set row weights
    row_w = get_uniform_row_weights(len(df))

    df_centered, mean, std = center(df)

    # Apply weights and compute svd
    Z = ((df_centered * _col_weights).T * row_w).T
    U, S, Vt = get_svd(Z, nf=nf, random_gen=random_gen)

    U = ((U.T) / np.sqrt(row_w)).T
    Vt = Vt / np.sqrt(_col_weights)

    explained_var, explained_var_ratio = get_explained_variance(S, df.shape[0], nf)

    # Retain only the nf higher singular values
    U = U[:, :nf]
    S = S[:nf]
    Vt = Vt[:nf, :]

    # we use the random generator to generate a new seed for the model
    new_seed = int(random_gen.integers(0, 2**32 - 1))
    model = Model(
        original_dtypes=df.dtypes,
        original_categorical=[],
        original_continuous=df.columns.to_list(),
        dummy_categorical=[],
        U=U,
        V=Vt,
        s=S,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(Vt.T),
        mean=mean,
        std=std,
        type="pca",
        is_fitted=True,
        nf=nf,
        column_weights=_col_weights,
        row_weights=row_w,
        modalities_types={},
        seed=new_seed,
    )

    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float64]] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a PCA model on data and return transformed data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
        coord: The transformed data of size (n, min(n,p))
        or (n, nf) if nf is specified.
    """
    model = fit(df, nf, col_weights, seed=seed)
    coord = transform(df, model)
    return coord, model


def center(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Center data and standardize it if scale. Compute mean and std values.

    Used as internal function during fit.

    **NB**: saiph.reduction.pca.scaler is better suited when a Model is already fitted.

    Parameters:
        df: DataFrame to center.

    Returns:
        df: The centered DataFrame.
        mean: Mean of the input dataframe.
        std: Standard deviation of the input dataframe.
    """
    df = df.copy()
    mean = np.mean(df, axis=0)
    df -= mean

    std = np.std(df, axis=0)

    # Remove zeros to avoid division by zero when a df contains a constant variable
    std_without_zeros = np.where(std <= sys.float_info.min, 1, std)
    df /= std_without_zeros

    return df, mean, std


def scaler(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Scale data using mean and std from model.

    Parameters:
        model: Model computed by fit.
        df: DataFrame to scale.

    Returns:
        df: The scaled DataFrame.
    """
    if model.std is None or model.mean is None:
        raise ValueError(
            "Expected model.std and model.mean to be not None,"
            f"got {model.std} and {model.mean} instead."
        )
    df_scaled = df.copy()

    df_scaled -= model.mean

    # Remove zeros to avoid division by zero when a df contains a constant variable
    std_without_zeros = pd.Series(
        np.where(model.std <= sys.float_info.min, 1, model.std), index=model.std.index
    )
    df_scaled /= std_without_zeros

    return df_scaled


def transform(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space of size (n, min(n,p))
        or (n, nf) if nf is specified.
    """
    df_scaled = scaler(model, df)
    coord = df_scaled @ model.V.T
    coord.columns = get_projected_column_names(model.nf)
    return coord


def reconstruct_df_from_model(model: Model) -> pd.DataFrame:
    """Reconstruct the original DataFrame from the model.

    Note: if nf < df.shape[1], reconstructed df will not be exactly the same.
    The more nf < df.shape[1], the more the reconstructed df will differ.
    the degree of difference is linked to the unused explained variance.

    Parameters:
        model: Model computed by fit.

    Returns:
        df: The reconstructed DataFrame.
    """
    # Extract the necessary components from the model
    if model.s is None or model.mean is None or model.std is None:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )
    U = model.U
    S = model.s
    V = model.V
    row_w = model.row_weights
    mean = model.mean
    std = model.std
    col_weights = model.column_weights

    # Construct the diagonal matrix of singular values
    Sigma = np.diag(S)

    # Reconstruct the weighted and scaled matrix Z
    Z = np.dot(U, np.dot(Sigma, V))

    # Undo the row and column weighting
    Z = Z / np.sqrt(row_w)[:, np.newaxis]
    Z = Z / np.sqrt(col_weights)

    # Rescale and shift to the original data distribution
    df_reconstructed = (Z * std.values) + mean.values

    # Convert to DataFrame and restore original column names and dtypes
    df_reconstructed = pd.DataFrame(df_reconstructed, columns=model.original_continuous)

    # Identify the columns that need to be converted to integers
    int_columns = model.original_dtypes[model.original_dtypes == "int"].index.tolist()

    # Round the values before because astype(int) truncates the decimals
    df_reconstructed[int_columns] = df_reconstructed[int_columns].round()

    # Convert the data types to the original types
    df_reconstructed = df_reconstructed.astype(model.original_dtypes)

    return df_reconstructed
