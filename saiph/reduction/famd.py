"""FAMD projection module."""
import sys
from itertools import chain, repeat
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction import DUMMIES_SEPARATOR
from saiph.reduction.utils.common import (
    column_multiplication,
    get_dummies_mapping,
    get_explained_variance,
    get_grouped_modality_values,
    get_modalities_types,
    get_projected_column_names,
    get_uniform_row_weights,
    row_division,
    row_multiplication,
)
from saiph.reduction.utils.svd import get_svd


def center(
    df: pd.DataFrame, quanti: List[str], quali: List[str]
) -> Tuple[
    pd.DataFrame, NDArray[np.float64], NDArray[np.float64], NDArray[Any], NDArray[Any]
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

    # Remove zeros to avoid division by zero when a df contains a constant variable
    std_without_zeros = np.where(std <= sys.float_info.min, 1, std)
    df_quanti /= std_without_zeros

    # scale the categorical data
    df_quali = pd.get_dummies(
        df[quali].astype("category"),
        prefix_sep=DUMMIES_SEPARATOR,
        dtype=np.uint8,
    )
    # .mean() is the same as counting 0s and 1s
    # This will only work if we stick with pd.get_dummies to encode the modalities
    # and we don't use `drop_first=True`
    prop = df_quali.mean()
    df_quali -= prop
    df_quali /= np.sqrt(prop)
    _modalities = df_quali.columns.values

    df_scale = pd.concat([df_quanti, df_quali], axis=1)

    return df_scale, mean, std, prop, _modalities


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float64]] = None,
    center: Callable[
        [pd.DataFrame, List[str], List[str]],
        Tuple[
            pd.DataFrame,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[Any],
            NDArray[Any],
        ],
    ] = center,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Model:
    """Fit a FAMD model on data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    nf = nf or min(pd.get_dummies(df).shape)
    _col_weights = np.ones(df.shape[1]) if col_weights is None else col_weights
    # If seed is None or int, we fit a Generator, else we use the one provided.
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )

    # Select the categorical and continuous columns
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.to_list()
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.to_list()
    dummy_categorical = pd.get_dummies(
        df[quali].astype("category"),
        prefix_sep=DUMMIES_SEPARATOR,
        dtype=np.uint8,
    ).columns.to_list()
    modalities_types = get_modalities_types(df[quali])

    row_w = get_uniform_row_weights(len(df))
    col_weights = _col_weights_compute(df, _col_weights, quanti, quali)

    df_scaled, mean, std, prop, _modalities = center(df, quanti, quali)

    # Apply the weights
    Z = df_scaled.multiply(col_weights).T.multiply(row_w).T

    # Compute the svd
    _U, S, _Vt = (
        get_svd(Z.todense(), nf=nf, random_gen=random_gen)
        if isinstance(Z, scipy.sparse.spmatrix)
        else get_svd(Z, nf=nf, random_gen=random_gen)
    )

    U = ((_U.T) / np.sqrt(row_w)).T
    Vt = _Vt / np.sqrt(col_weights)

    explained_var, explained_var_ratio = get_explained_variance(S, df.shape[0], nf)

    # Retain only the nf higher singular values
    U = U[:, :nf]
    S = S[:nf]
    Vt = Vt[:nf, :]
    # we use the random generator to generate a new seed for the model
    new_seed = int(random_gen.integers(0, 2**32 - 1))
    model = Model(
        original_dtypes=df.dtypes,
        original_categorical=quali,
        original_continuous=quanti,
        dummy_categorical=dummy_categorical,
        U=U,
        V=Vt,
        s=S,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(Vt.T),
        mean=mean,
        std=std,
        prop=prop,
        _modalities=_modalities,
        type="famd",
        is_fitted=True,
        nf=nf,
        column_weights=col_weights,
        row_weights=row_w,
        modalities_types=modalities_types,
        seed=new_seed,
    )

    return model


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
        coord: The transformed data of size (n, min(n,p))
        or (n, nf) if nf is specified.
        model: The model for transforming new data.
    """
    # If seed is None or int, we fit a Generator, else we use the one provided.
    random_gen = (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )
    model = fit(df, nf, col_weights, seed=random_gen)
    coord = transform(df, model)
    return coord, model


def _col_weights_compute(
    df: pd.DataFrame, col_weights: NDArray[Any], quanti: List[int], quali: List[int]
) -> NDArray[Any]:
    """Calculate weights for columns given what weights the user gave."""
    # Set the columns and row weights
    weight_df = pd.DataFrame([col_weights], columns=df.columns)
    weight_quanti = weight_df[quanti]
    weight_quali = weight_df[quali]

    # Get the number of modality for each quali variable
    modality_numbers = []
    for column in weight_quali.columns:
        modality_numbers += [len(df[column].unique())]

    # Set weight vector for categorical columns
    weight_quali_rep = list(
        chain.from_iterable(
            repeat(i, j) for i, j in zip(list(weight_quali.iloc[0]), modality_numbers)
        )
    )

    _col_weights: NDArray[Any] = np.array(
        list(weight_quanti.iloc[0]) + weight_quali_rep
    )

    return _col_weights


def scaler(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Scale data using mean, std, modalities and proportions of each categorical from model.

    Parameters:
        model: Model computed by fit.
        df: DataFrame to scale.

    Returns:
        df_scaled: The scaled DataFrame.
    """
    if model.mean is None or model.std is None:
        raise ValueError(
            "Expected model to have mean and std attributes,",
            f"got {model.mean} and {model.std} instead.",
        )
    model.prop = cast(pd.Series, model.prop)

    df_quanti = df[model.original_continuous]

    # Remove zeros to avoid division by zero when a df contains a constant variable
    std_without_zeros = pd.Series(
        np.where(model.std <= sys.float_info.min, 1, model.std), index=model.std.index
    )
    df_quanti = (df_quanti - model.mean) / std_without_zeros

    # scale
    df_quali = pd.get_dummies(
        df[model.original_categorical].astype("category"),
        prefix_sep=DUMMIES_SEPARATOR,
        dtype=np.uint8,
    )
    # Here we add a column with 0 if the modality is not present in the dataset but
    # was used to train the saiph model
    if model._modalities is not None:
        for mod in model._modalities:
            if mod not in df_quali:
                df_quali[mod] = 0
    df_quali = df_quali[model._modalities]
    df_quali = (df_quali - model.prop) / np.sqrt(model.prop)

    df_scaled = pd.concat([df_quanti, df_quali], axis=1)
    return df_scaled


def transform(
    df: pd.DataFrame,
    model: Model,
    *,
    scaler: Callable[[Model, pd.DataFrame], pd.DataFrame] = scaler,
) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space of size (n, min(n,p))
        or (n, nf) if nf is specified.
    """
    df_scaled = scaler(model, df)
    coord = pd.DataFrame(df_scaled @ model.V.T)

    coord.columns = get_projected_column_names(len(coord.columns))
    return coord


def stats(model: Model, df: pd.DataFrame, explode: bool = False) -> Model:
    """Compute contributions and cos2.

    Parameters:
        model: Model computed by fit.
        df: dataframe to compute statistics from
        explode: whether to split the contributions of each modality (True)
            or sum them as the contribution of the whole variable (False)

    Returns:
        model: model populated with contribution and cos2.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    contributions, cos2 = get_variable_contributions(model, df, explode=explode)
    model.contributions = contributions
    model.cos2 = cos2
    return model


def get_variable_contributions(
    model: Model, df: pd.DataFrame, explode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the contributions of the `df` variables within the fitted space.

    Parameters:
        model: Model computed by fit.
        df: dataframe to compute contributions from
        explode: whether to split the contributions of each modality (True)
            or sum them as the contribution of the whole variable (False)

    Returns:
        tuple of contributions and cos2.
    """
    scaled_df = pd.DataFrame(scaler(model, df))
    weighted_df = column_multiplication(scaled_df, np.sqrt(model.column_weights))
    weighted_df = row_multiplication(weighted_df, np.sqrt(model.row_weights))

    min_nf = min(min(weighted_df.shape), model.nf)
    s, U, eig = _compute_svd(model, weighted_df, min_nf=min_nf)
    contributions = _compute_contributions(
        model, s, U, eig, min_nf, scaled_df.columns, explode=explode
    )

    continuous_cos2 = compute_continuous_cos2(model, scaled_df, min_nf, s, U)
    categorical_cos2 = compute_categorical_cos2(model, df, min_nf)
    combined_cos2 = pd.concat([continuous_cos2, categorical_cos2])

    return contributions, combined_cos2


def _compute_contributions(
    model: Model,
    s: NDArray[np.float64],
    U: NDArray[np.float64],
    eig: NDArray[np.float64],
    min_nf: int,
    column_names: List[str],
    *,
    explode: bool = True,
) -> pd.DataFrame:
    # compute the contribution
    # Remove zeros to avoid division by zero when a df contains a constant variable
    eig_without_zeros = np.where(eig <= sys.float_info.min, 1, eig)
    raw_contributions = (U * s) ** 2 / eig_without_zeros

    raw_contributions = raw_contributions * model.column_weights[:, np.newaxis]
    summed_contributions: NDArray[np.float64] = np.array(raw_contributions.sum(axis=0))

    # Remove zeros to avoid division by zero when a df contains a constant variable
    summed_contributions_without_zeros = np.where(
        summed_contributions <= sys.float_info.min, 1, summed_contributions
    )
    raw_contributions /= summed_contributions_without_zeros
    raw_contributions *= 100

    contributions = pd.DataFrame(
        raw_contributions,
        index=column_names,
        columns=get_projected_column_names(min_nf),
    )

    if explode:
        return contributions

    mapping = get_dummies_mapping(model.original_categorical, model.dummy_categorical)

    contributions = get_grouped_modality_values(mapping, contributions)

    return contributions


def compute_categorical_cos2(
    model: Model, df: pd.DataFrame, min_nf: int
) -> pd.DataFrame:
    """Compute the cos2 statistic for categorical variables.

    Parameters
    ----------
    model :
        model
    df :
        dataframe
    min_nf :
        number of degrees of freedom

    Returns
    -------
        dataframe of categorical cos2
    """
    if model.U is not None and model.s is not None:
        model_coords = pd.DataFrame(
            model.U[:, :min_nf] * model.s[:min_nf],
            columns=get_projected_column_names(min_nf),
        )

    mapping = get_dummies_mapping(model.original_categorical, model.dummy_categorical)
    dummy = pd.get_dummies(
        df[model.original_categorical].astype("category"),
        prefix_sep=DUMMIES_SEPARATOR,
        dtype=np.uint8,
    )
    # Compute the categorical cos2 for each original column
    all_category_cos = {}
    for original_col, dummy_columns in mapping.items():
        # FIXME: Kept this for legacy: Why - 1 ? nb_modalities += [nb_dummy_columns - 1] #
        # for each dimension
        all_category_cos[original_col] = _compute_cos2_single_category(
            dummy[dummy_columns], model, model_coords
        )

    # FIXME: Why - 1 ?
    nb_modalities = df[model.original_categorical].nunique() - 1

    categorical_cos2 = pd.DataFrame.from_dict(
        data=all_category_cos,
        orient="index",
        columns=get_projected_column_names(min_nf),
    )

    categorical_cos2 = row_division(categorical_cos2**2, nb_modalities)

    return categorical_cos2


def compute_continuous_cos2(
    model: Model,
    scaled_df: pd.DataFrame,
    min_nf: int,
    s: NDArray[np.float64],
    U: NDArray[np.float64],
) -> pd.DataFrame:
    squared_values: NDArray[np.float64] = scaled_df.values**2

    weighted_values = squared_values * model.row_weights[:, np.newaxis]

    dist2 = np.sum(weighted_values, axis=0)
    dist2 = np.where(np.abs(dist2 - 1) < 0.001, 1, np.sqrt(dist2))

    dist2_without_zeros = np.where(dist2[:, np.newaxis] == 0, 1, dist2[:, np.newaxis])
    cos2 = np.divide(U * s, dist2_without_zeros) ** 2
    # FIXME: Why original_continuous, we have been computing cos2 for all the dummies for nothing?
    continuous_cos2 = pd.DataFrame(
        cos2[: len(model.original_continuous)],  # only keep continuous components
        index=model.original_continuous,
        columns=get_projected_column_names(min_nf),
    )
    continuous_cos2 = continuous_cos2**2
    return continuous_cos2


def _compute_svd(
    model: Model, weighted: NDArray[np.float64], min_nf: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    U, s, V = get_svd(weighted.T, svd_flip=False)

    # Only keep first nf components
    U = U[:, :min_nf]
    V = V.T[:, :min_nf]
    s = s[:min_nf]
    weights = model.column_weights[:min_nf]

    # FIXME: kept this here for legacy, don't know why we were doing it: U, V = V, U

    sign = np.sign(np.sum(U))
    signed_U = column_multiplication(pd.DataFrame(U), sign).values

    # Divide diagonal values by weights
    min_shape = min(signed_U.shape)
    diagonal = np.diag_indices_from(signed_U[:min_shape, :min_shape])
    weighted_U = signed_U
    weighted_U[diagonal] = np.true_divide(signed_U[diagonal], weights)

    eigenvalues = np.power(s, 2)
    return s, weighted_U, eigenvalues


def _compute_cos2_single_category(
    single_category_df: pd.DataFrame, model: Model, coords: pd.DataFrame
) -> NDArray[np.float64]:
    """Compute cos2 for a single original category.

    Parameters
    ----------
    single_category_df:
        dummy dataframe of a single original category
    model :
        _description_
    coords :
        projections of the data used to created the axes

    Returns
    -------
        cos2 of the single category
    """
    _, n_cols = single_category_df.shape
    cos2 = []
    for coord_col in coords.columns:
        weighted_coord = coords[coord_col].values * model.row_weights
        p_values = np.zeros(n_cols)

        for i, col in enumerate(single_category_df.columns):
            dummy_values = single_category_df[col].values
            if model.prop is not None:
                p_values[i] = (dummy_values * weighted_coord).sum() ** 2 / model.prop[
                    col
                ]

        p = p_values.sum()
        cos2.append(p)

    all_weighted_coords = (coords.values**2).T * model.row_weights
    summed_weights = all_weighted_coords.sum(axis=1)

    # Remove zeros to avoid division by zero when a df contains a constant variable
    summed_weights_without_zeros = np.where(
        summed_weights <= sys.float_info.min, 1, summed_weights
    )
    single_category_cos2: NDArray[np.float64] = (
        np.array(cos2) / summed_weights_without_zeros
    )
    return single_category_cos2


def reconstruct_df_from_model(model: Model) -> pd.DataFrame:
    """Reconstruct a DataFrame from a fitted model.

    Note: if nf < df.shape[1], reconstructed df will not be exactly the same.
    The more nf < df.shape[1], the more the reconstructed df will differ.
    the degree of difference is linked to the unused explained variance.

    Parameters:
        model: Model computed by fit.

    Returns:
        df: The reconstructed DataFrame.
    """
    # Extract the necessary components from the model
    if model.s is None or model.mean is None or model.std is None or model.prop is None:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )
    U = model.U
    S = model.s
    V = model.V
    row_w = model.row_weights
    col_weights = model.column_weights
    mean = model.mean.values
    std = model.std.values
    prop = model.prop.values
    _modalities = model._modalities
    quanti = model.original_continuous
    quali = model.original_categorical

    # Construct the diagonal matrix of singular values
    Sigma = np.diag(S)

    # Reconstruct the weighted and scaled matrix Z
    Z = np.dot(U, np.dot(Sigma, V))

    # Undo the row and column weighting
    Z = Z / np.sqrt(row_w)[:, np.newaxis]
    Z = Z / np.sqrt(col_weights)

    # Split Z back into quantitative and qualitative parts
    n_quanti = len(quanti)
    df_quanti_scaled = Z[:, :n_quanti]
    df_quali_scaled = Z[:, n_quanti:]

    # Reverse the scaling for quantitative variables
    df_quanti = df_quanti_scaled * std + mean

    # Reverse the scaling for qualitative variables
    df_quali = df_quali_scaled * np.sqrt(prop) + prop

    # Combine quantitative and qualitative data
    df_quali = pd.DataFrame(df_quali, columns=_modalities)
    df_quanti = pd.DataFrame(df_quanti, columns=quanti)
    df_reconstructed = pd.concat([df_quanti, df_quali], axis=1)

    # Reverse the dummy encoding for categorical variables
    for var in quali:
        prefix = var + DUMMIES_SEPARATOR
        dummies = [col for col in df_reconstructed.columns if col.startswith(prefix)]
        df_reconstructed[var] = (
            df_reconstructed[dummies]
            .idxmax(axis=1)
            .apply(lambda x: x.split(DUMMIES_SEPARATOR)[1])
        )
        df_reconstructed.drop(columns=dummies, inplace=True)

    # Ensure the column order matches the original dataframe
    df_reconstructed = df_reconstructed[model.original_dtypes.index]

    # Identify the columns that need to be converted to integers
    int_columns = model.original_dtypes[model.original_dtypes == "int"].index.tolist()

    # Round the values before because astype(int) truncates the decimals
    df_reconstructed[int_columns] = df_reconstructed[int_columns].round()

    # Convert the data types to the original types
    df_reconstructed = df_reconstructed.astype(model.original_dtypes)

    return df_reconstructed
