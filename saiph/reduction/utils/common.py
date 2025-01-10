from itertools import repeat
from typing import Any, Dict, List, OrderedDict, Tuple

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from toolz import concat

from saiph.reduction import DUMMIES_SEPARATOR


def get_projected_column_names(n: int) -> List[str]:
    return [f"Dim. {i + 1}" for i in range(n)]


def get_uniform_row_weights(n: int) -> NDArray[np.float64]:
    return np.array([k for k in repeat(1 / n, n)])


def row_multiplication(df: pd.DataFrame, arr: NDArray[Any]) -> pd.DataFrame:
    """Multiply each row of `df` with the corresponding value in `arr`."""
    return df.multiply(arr, axis="rows")


def column_multiplication(df: pd.DataFrame, arr: NDArray[Any]) -> pd.DataFrame:
    """Multiply each column of `df` with the corresponding value in `arr`."""
    return df.multiply(arr, axis="columns")


def row_division(df: pd.DataFrame, arr: NDArray[Any]) -> pd.DataFrame:
    """Divide each row of `df` with the corresponding value in `arr`."""
    return df.divide(arr, axis="rows")


def diag(arr: NDArray[Any], use_scipy: bool = False) -> NDArray[Any]:
    if use_scipy:
        return scipy.sparse.diags(arr)  # type: ignore
    else:
        return np.diag(arr)


def get_explained_variance(
    s: NDArray[np.float64],
    nb_individuals: int,
    nf: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    all_variance = (s**2) / (nb_individuals - 1)
    variance = all_variance[:nf]

    # We divide by the all_variance and not variance because
    # if nf < len(all_variance), we shouldn't sum up to 100%
    variance_sum = all_variance.sum()

    variance_ratio = (
        variance / variance_sum if variance_sum != 0 else np.full_like(variance, np.nan)
    )

    return variance, variance_ratio


def get_modalities_types(df: pd.DataFrame) -> Dict[str, str]:
    modalities_types = {col: get_type_as_string(df.loc[0, col]) for col in df.columns}
    return modalities_types


def get_dummies_mapping(
    columns: List[str], dummy_columns: List[str]
) -> Dict[str, List[str]]:
    """Get mapping between original column and all dummy columns."""
    return OrderedDict(
        {
            col: list(
                filter(
                    lambda c: c.startswith(f"{col}{DUMMIES_SEPARATOR}"), dummy_columns
                )
            )
            for col in columns
        }
    )


TYPES = {
    int: "int",
    np.int_: "int",
    float: "float",
    np.float64: "float",
    str: "string",
    np.str_: "string",
    bool: "bool",
    np.bool_: "bool",
}


def get_type_as_string(value: Any) -> str:
    """Return string of type."""
    return TYPES[type(value)]


def get_grouped_modality_values(
    mapping: Dict[str, List[str]], to_group: pd.DataFrame
) -> pd.DataFrame:
    """Get the sum of the values of modalities into the category.

    Parameters
    ----------
    mapping :
        mapping between categorical columns and their dummy equivalent
    to_group :
        dataframe from which to sum the values of modalities, which are
        passed as the index.

    Returns
    -------
        a dataframe with the categorical variables without the dummies
    """
    if not mapping:  # We have no mapping, we have no categorical variables
        return to_group
    grouped_contributions = {}
    for original_col, dummy_columns in mapping.items():
        grouped_contributions[original_col] = to_group.loc[dummy_columns].sum(axis=0)

    grouped_contributions = pd.DataFrame.from_dict(
        data=grouped_contributions,
        orient="index",
        columns=to_group.columns,
    )

    # FIXME: Do not modify the order or columns
    # https://github.com/octopize/saiph/issues/40
    grouped = pd.concat([to_group, grouped_contributions])
    grouped = grouped.drop(concat(mapping.values()))
    return grouped
