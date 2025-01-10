from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from toolz import concat

from saiph.reduction import DUMMIES_SEPARATOR
from saiph.reduction.utils.common import (
    column_multiplication,
    get_dummies_mapping,
    get_explained_variance,
    get_grouped_modality_values,
    row_division,
    row_multiplication,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame([[1, 10], [2, 20]])


def test_row_multiplication(df: pd.DataFrame) -> None:
    expected = pd.DataFrame([[1, 10], [4, 40]])

    result = row_multiplication(df, np.array([1, 2]))

    assert_frame_equal(result, expected)


def test_column_multiplication(df: pd.DataFrame) -> None:
    expected = pd.DataFrame([[1, 20], [2, 40]])

    result = column_multiplication(df, np.array([1, 2]))
    assert_frame_equal(result, expected)


def test_row_division(df: pd.DataFrame) -> None:
    expected = pd.DataFrame([[1, 10], [1, 10]], dtype=float)

    result = row_division(df, np.array([1, 2]))

    assert_frame_equal(result, expected)


def test_get_dummies_mapping(
    quali_df: pd.DataFrame, mapping: Dict[str, List[str]]
) -> None:
    dummy_columns = pd.get_dummies(
        quali_df,
        prefix_sep=DUMMIES_SEPARATOR,
        dtype=np.uint8,
    ).columns
    result = get_dummies_mapping(quali_df.columns, dummy_columns)

    assert result == mapping


def test_get_grouped_modality_values(mapping: Dict[str, List[str]]) -> None:
    """Verify that grouping modalities returns the correct groupings."""
    df = pd.DataFrame.from_dict(
        {col: [10] for col in concat(mapping.values())}, orient="index"
    )
    expected = pd.DataFrame.from_dict(
        {col: [20] for col in mapping.keys()}, orient="index"
    )
    grouped_df = get_grouped_modality_values(mapping, df)

    assert_frame_equal(grouped_df.sort_index(), expected.sort_index())


@pytest.mark.parametrize(
    "s,expected_variance,expected_ratio",
    [
        ([10, 10, 10], [50.0, 50.0], [1 / 3, 1 / 3]),  # nf < len(s), we don't get 100%.
        ([0.0, 0.0, 0.0], [0.0, 0.0], [np.nan, np.nan]),
    ],
)
def test_get_explained_variance_returns_correct_variance_and_ratio(
    s: Any, expected_variance: Any, expected_ratio: Any
) -> None:
    variance, ratio = get_explained_variance(np.array(s), nb_individuals=3, nf=2)
    np.testing.assert_array_equal(variance, expected_variance)
    np.testing.assert_array_equal(ratio, expected_ratio)
