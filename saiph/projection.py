"""Project any dataframe and compute stats."""
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.exception import ColumnsNotFoundError, InvalidParameterException
from saiph.models import Model
from saiph.reduction import DUMMIES_SEPARATOR, famd, famd_sparse, mca, pca
from saiph.reduction.utils.common import get_projected_column_names


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[Dict[str, Union[int, float]]] = None,
    sparse: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Model:
    """Fit a PCA, MCA or FAMD model on data, imputing what has to be used.

    Datetimes must be stored as numbers of seconds since epoch.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: None, which uses all columns.
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes).
            default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    if nf is not None and (nf <= 0 or nf > min(pd.get_dummies(df).shape)):
        raise InvalidParameterException(
            "Expected number of components to be in "
            f"0 < 'nf' <= {min(pd.get_dummies(df).shape)}, got {nf} instead."
        )

    if col_weights is not None:
        unknown_variables = list(
            filter(lambda c: c not in df.columns, col_weights.keys())
        )
        if unknown_variables:
            raise InvalidParameterException(
                "Expected valid variable name(s) in 'col_weights', "
                f"got {unknown_variables} instead."
            )

    _nf = nf if nf else min(pd.get_dummies(df, prefix_sep=DUMMIES_SEPARATOR).shape)
    # If seed is None or int, we fit a Generator, else we use the one provided.
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )

    # Convert col weights from dict to ndarray
    _col_weights: NDArray[np.float64] = np.ones(df.shape[1])

    if col_weights is not None:
        for col in col_weights:
            _col_weights[df.columns.get_loc(col)] = col_weights[col]

    # Check column types
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.values
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.values

    # Specify the correct function
    if quali.size == 0:
        _fit = pca.fit
    elif quanti.size == 0:
        _fit = mca.fit
    elif sparse:
        _fit = famd_sparse.fit
    else:
        # TODO if famd_sparse is removed:
        # remove param center from famd.fit and no need to ignore type
        _fit = famd.fit  # type: ignore

    model = _fit(df, _nf, _col_weights, seed=random_gen)

    if quanti.size == 0:
        model.D_c = cast(NDArray[np.float64], model.D_c)
        model.variable_coord = pd.DataFrame(model.D_c @ model.V.T)
    else:
        model.variable_coord = pd.DataFrame(model.V.T)
    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[Dict[str, Union[int, float]]] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a PCA, MCA or FAMD model on data, imputing what has to be used.

    Datetimes must be stored as numbers of seconds since epoch.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: 'all'
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes).
            default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    # If seed is None or int, we fit a Generator, else we use the one provided.
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )
    model = fit(df, nf=nf, col_weights=col_weights, seed=random_gen)
    coord = transform(df, model)
    return coord, model


def stats(model: Model, df: pd.DataFrame, explode: bool = False) -> Model:
    """Compute the contributions and cos2.

    Parameters:
        model: Model computed by fit.
        df: original dataframe
        explode: whether to split the contributions of each modality (True)
            or sum them as the contribution of the whole variable (False).
            Only valid for categorical variables.

    Returns:
        model: model populated with contribution.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    model.correlations = get_variable_correlation(model, df)
    model.variable_coord.columns = get_projected_column_names(
        model.variable_coord.shape[1]
    )
    model.variable_coord.index = list(model.correlations.index)

    has_some_quanti = (
        model.original_continuous is not None and len(model.original_continuous) != 0
    )
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )

    if not has_some_quali:
        model.cos2 = model.correlations**2
        model.contributions = model.cos2.div(model.cos2.sum(axis=0), axis=1).mul(100)
    elif not has_some_quanti:
        model = mca.stats(model, df, explode=explode)
    else:
        model = famd.stats(model, df, explode=explode)

    return model


def get_variable_contributions(
    model: Model, df: pd.DataFrame, explode: bool = False
) -> pd.DataFrame:
    """Compute the contributions of the `df` variables within the fitted space.

    Parameters:
        model: Model computed by fit.
        df: dataframe to compute contributions from
        explode: whether to split the contributions of each modality (True)
            or sum them as the contribution of the whole variable (False)

    Returns:
        contributions
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    has_some_quanti = (
        model.original_continuous is not None and len(model.original_continuous) != 0
    )
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )

    if not has_some_quali:
        correlations = get_variable_correlation(model, df)
        cos2 = correlations**2
        contributions = cos2.div(cos2.sum(axis=0), axis=1).mul(100)
        contributions = contributions.set_index(df.columns)
        return contributions

    if not has_some_quanti:
        return mca.get_variable_contributions(model, df, explode=explode)

    contributions, _ = famd.get_variable_contributions(model, df, explode=explode)
    return contributions


def transform(df: pd.DataFrame, model: Model, *, sparse: bool = False) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted."
            "Call fit() to create a Model instance before calling transform()."
        )

    # Check that all the columns of the df is in the model
    model_columns = model.original_continuous + model.original_categorical
    df_columns = df.columns.tolist()
    if sorted(df_columns) != sorted(model_columns):
        difference = set(df_columns) - set(model_columns)
        raise ColumnsNotFoundError(
            "Expected columns to be the same as the ones used in the model."
            f"Got {difference}."
        )

    if len(model.original_categorical) == 0:
        return pca.transform(df, model)

    if len(model.original_continuous) == 0:
        return mca.transform(df, model)

    if sparse:
        return famd_sparse.transform(df, model)

    return famd.transform(df, model)


def get_variable_correlation(
    model: Model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the correlation between the axis and the variables.

    Parameters:
        model: the model
        df: dataframe

    Returns:
        cor: correlations between the axis and the variables
    """
    # select columns and project data
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )
    df_quanti = df[model.original_continuous]
    coord = transform(df, model)  # transform to be
    if has_some_quali:
        df_quali = pd.get_dummies(
            df[model.original_categorical].astype("category"),
            prefix_sep=DUMMIES_SEPARATOR,
            dtype=np.uint8,
        )
        bind = pd.concat([df_quanti, df_quali], axis=1)
    else:
        bind = df_quanti

    concat = pd.concat([bind, coord], axis=1, keys=["bind", "coord"])
    cor = pd.DataFrame(np.corrcoef(concat, rowvar=False), columns=concat.columns).loc[
        0 : len(bind.columns) - 1,
        "coord",
    ]
    return cor


def get_reconstructed_df_from_model(model: Model) -> pd.DataFrame:
    """Reconstruct the original DataFrame from the model.

    Note: if nf < df.shape[1], reconstructed df will not be exactly the same.
    The more nf < df.shape[1], the more the reconstructed df will differ.
    the degree of difference is linked to the unused explained variance.

    Parameters:
        model: Model computed by fit.

    Returns:
        df: The reconstructed DataFrame.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted."
            "Call fit() to create a Model instance before calling transform()."
        )

    if len(model.original_categorical) == 0:
        return pca.reconstruct_df_from_model(model)

    if len(model.original_continuous) == 0:
        return mca.reconstruct_df_from_model(model)

    return famd.reconstruct_df_from_model(model)
