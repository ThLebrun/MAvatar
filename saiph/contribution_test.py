import numpy as np
import pandas as pd
import pytest

from saiph.contribution import filter_variable_contribution


@pytest.fixture
def contributions() -> pd.DataFrame:
    df = pd.DataFrame(
        data=[[1, 0.01], [2, 1]],
        index=["Var. 1", "Var. 2"],
        columns=["Dim. 1", "Dim. 2"],
    )

    return df


def test_filter_variable_contribution_is_sorted(contributions: pd.DataFrame) -> None:
    """Verify that we get sorted descending values."""
    result = filter_variable_contribution(
        contributions=contributions, dim=2, max_var=None, min_contrib=None
    )
    np.testing.assert_array_equal(result.values, [1, 0.01])
    np.testing.assert_array_equal(result.index.values, ["Var. 2", "Var. 1"])


def test_filter_variable_contribution_filter_by_min_contrib(
    contributions: pd.DataFrame,
) -> None:
    """Verify that we can filter by minimum contribution."""
    result = filter_variable_contribution(
        contributions=contributions, dim=2, max_var=None, min_contrib=0.5
    )
    np.testing.assert_array_equal(result.values, [1])


def test_filter_variable_contribution_filter_by_max_var(
    contributions: pd.DataFrame,
) -> None:
    """Verify that we can filter by maximum number of variables."""
    result = filter_variable_contribution(contributions=contributions, dim=2, max_var=1)
    np.testing.assert_array_equal(result.values, [1])
