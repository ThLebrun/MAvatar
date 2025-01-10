import sys
from typing import Optional

import pandas as pd


def filter_variable_contribution(
    contributions: pd.DataFrame,
    dim: int = 1,
    *,
    max_var: Optional[int] = 10,
    min_contrib: Optional[float] = 0.1,
) -> pd.Series:
    """Get sorted and filtered variable contributions for a given dimension.

    Parameters:
        contributions: Variable contributions of the model, per dimension.
        dim: Dimension to plot. default: 1
        max_var : Maximum number of variables to plot. default: 10
        min_contrib : Minimum contribution threshold for the variable
                    contributions to be displayed. default: 0.1

    Returns:
        selected: Contributions of the specified dimension, sorted by descending importance
            and filtered by the given values of `max_var` and `min_contrib`.
    """
    nb_variables = max_var or len(contributions)
    min_contrib = min_contrib or sys.float_info.min
    label = contributions.columns[dim - 1]  # Dimensions starts at 1
    selected = (
        contributions[label]
        .loc[lambda v: v > min_contrib]
        .head(nb_variables)
        .sort_values(ascending=False)
    )
    return selected
