"""Inverse transform coordinates."""
import ast
from typing import Dict, List, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction import DUMMIES_SEPARATOR
from saiph.reduction.utils.common import get_dummies_mapping


def inverse_transform(
    coord: pd.DataFrame,
    model: Model,
    *,
    use_max_modalities: bool = True,
) -> pd.DataFrame:
    """Return original format dataframe from coordinates.

    Parameters:
        coord: coord of individuals to reverse transform
        model: model used for projection
        use_max_modalities: for each variable, it assigns to the individual
            the modality with the highest proportion (True)
            or a random modality weighted by their proportion (False). default: True

    Returns:
        inverse: coordinates transformed into original space.
            Retains shape, encoding and structure.
    """
    random_gen = np.random.default_rng(model.seed)

    # Get back scaled_values from coord with inverse matrix operation
    scaled_values = pd.DataFrame(coord @ np.linalg.pinv(model.V.T))
    # get number of continuous variables
    nb_quanti = len(model.original_continuous)

    # separate quanti from quali
    scaled_values_quanti = scaled_values.iloc[:, :nb_quanti]
    scaled_values_quanti.columns = model.original_continuous

    scaled_values_quali = scaled_values.iloc[:, nb_quanti:]
    scaled_values_quali.columns = model.dummy_categorical

    # Descale regarding projection type
    # FAMD
    if model.type == "famd":
        model.prop = cast(pd.Series, model.prop)
        descaled_values_quanti = (scaled_values_quanti * model.std) + model.mean
        descaled_values_quali = (scaled_values_quali * np.sqrt(model.prop)) + model.prop

        del scaled_values_quali
        del scaled_values_quanti
        undummy = undummify(
            descaled_values_quali,
            get_dummies_mapping(model.original_categorical, model.dummy_categorical),
            use_max_modalities=use_max_modalities,
            random_gen=random_gen,
        )
        inverse = pd.concat([descaled_values_quanti, undummy], axis=1).round(12)

    # PCA
    elif model.type == "pca":
        descaled_values_quanti = (scaled_values_quanti * model.std) + model.mean
        inverse = descaled_values_quanti.round(12)
        del scaled_values_quali
        del scaled_values_quanti

    # MCA
    else:
        del scaled_values_quali
        del scaled_values_quanti
        model.D_c = cast(NDArray[np.float64], model.D_c)

        # As we are not scaling MCA such as FAMD categorical, the descale is
        # not the same. Doing the same as FAMD is incoherent.
        inverse_data = coord @ (model.D_c @ model.V.T).T
        inverse_coord_quali = inverse_data.set_axis(
            model.dummy_categorical, axis="columns"
        )

        descaled_values_quali = inverse_coord_quali.divide(model.dummies_col_prop)
        inverse = undummify(
            descaled_values_quali,
            get_dummies_mapping(model.original_categorical, model.dummy_categorical),
            use_max_modalities=use_max_modalities,
            random_gen=random_gen,
        )
    # Cast columns to same type as input
    for name, dtype in model.original_dtypes.items():
        # Can create a bug if a column is object but contains int and float values,
        # first, we force the value type of the first value of the original df
        if dtype in ["object", "category"]:
            if model.modalities_types[name] == "bool":
                inverse[name] = [ast.literal_eval(ele) for ele in inverse[name]]
            else:
                inverse[name] = inverse[name].astype(model.modalities_types[name])

        inverse[name] = inverse[name].astype(dtype)

    # reorder columns
    return inverse[model.original_dtypes.index]


def undummify(
    dummy_df: pd.DataFrame,
    dummies_mapping: Dict[str, List[str]],
    *,
    use_max_modalities: bool = True,
    random_gen: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """Return undummified dataframe from the dummy dataframe.

    Parameters:
        dummy_df: dummy df of categorical variables
        dummies_mapping: mapping between categorical columns and dummies columns.
        use_max_modalities: True to select the modality with the highest probability.
                            False for a weighted random selection. default: True
        seed: seed to fix randomness if use_max_modalities = False. default: None

    Returns:
        inverse_quali: undummify df of categorical variable
    """
    inverse_quali = pd.DataFrame()

    def get_modality_from_dummy_variable(string: str, original_column: str) -> str:
        return string.removeprefix(original_column + DUMMIES_SEPARATOR)

    for original_column, dummy_columns in dummies_mapping.items():
        # Handle a single category with all the possible modalities
        single_category = dummy_df[dummy_columns]
        if use_max_modalities:
            # select modalities with highest probability
            chosen_modalities = single_category.idxmax(axis="columns")
        else:
            chosen_modalities = get_random_weighted_columns(single_category, random_gen)
        inverse_quali[original_column] = list(
            map(
                lambda x: get_modality_from_dummy_variable(x, original_column),
                chosen_modalities,
            )
        )

    return inverse_quali


def get_random_weighted_columns(
    df: pd.DataFrame, random_gen: np.random.Generator
) -> pd.Series:
    """Randomly select column labels weighted by proportions.

    Parameters:
        df : dataframe containing proportions
        random_gen: random generator

    Returns:
        column_labels: selected column labels
    """
    # Example for 1 row:  [0.1, 0.3, 0.6] --> [0.1, 0.4, 1.0]
    # probabilities needs to be normalized before draw
    normalized_df = df.div(df.sum(axis=1), axis=0)
    cum_probability = normalized_df.cumsum(axis=1)
    random_probability = random_gen.random((cum_probability.shape[0], 1))

    # [0.342] < [0.1, 0.4, 1.0] --> [False, True, True] --> idx: 1
    column_labels = (random_probability < cum_probability).idxmax(axis=1)

    return column_labels
