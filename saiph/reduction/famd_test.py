from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from saiph.reduction import DUMMIES_SEPARATOR
from saiph.reduction.famd import (
    center,
    fit,
    fit_transform,
    get_variable_contributions,
    reconstruct_df_from_model,
    scaler,
    transform,
)
from saiph.reduction.pca import center as center_pca
from saiph.reduction.pca import fit_transform as fit_pca
from saiph.reduction.pca import scaler as scaler_pca
from saiph.reduction.utils.common import get_projected_column_names


def test_fit_mix(mixed_df2: pd.DataFrame) -> None:
    df = mixed_df2.iloc[:, 0:3]

    result, model = fit_transform(df)
    model.mean = cast(pd.Series, model.mean)
    model.std = cast(pd.Series, model.std)
    model.s = cast(NDArray[np.float64], model.s)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [-1.73, 1.73],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v: NDArray[np.float64] = np.array(
        [
            [0.57735, 0.408248, -0.408248, -0.408248, 0.408248],
            [0.816497, -0.288675, 0.288675, 0.288675, -0.288675],
        ]
    )
    expected_s: NDArray[np.float64] = np.array([1.224745e00, 0.0])
    expected_u: NDArray[np.float64] = np.array([[-1.0, 1.0], [1.0, 1.0]])
    expected_explained_var: NDArray[np.float64] = np.array([1.5, 0.0])
    expected_explained_var_ratio: NDArray[np.float64] = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.s, expected_s, atol=0.01)
    assert_allclose(model.U, expected_u, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    assert_allclose(model.variable_coord, model.V.T)
    assert np.array_equal(
        model._modalities,  # type: ignore
        [
            f"tool{DUMMIES_SEPARATOR}hammer",
            f"tool{DUMMIES_SEPARATOR}toaster",
            f"score{DUMMIES_SEPARATOR}aa",
            f"score{DUMMIES_SEPARATOR}ab",
        ],
    )
    # Pertinent ?
    # assert_allclose(model.D_c,np.array([[2.0, 0.0, 0.0],
    # [0.0, 2.0, 0.0], [0.0, 0.0, 1.41421356]]), atol=0.01)
    assert model.D_c is None

    assert_allclose(model.mean, np.array(2.5))
    assert_allclose(model.std, np.array(1.5))
    if model.prop is not None:
        assert_series_equal(
            model.prop.reset_index(drop=True),
            pd.Series([0.5, 0.5, 0.5, 0.5]).reset_index(drop=True),
            atol=0.01,
        )
    assert np.array_equal(
        model._modalities,  # type: ignore
        [
            f"tool{DUMMIES_SEPARATOR}hammer",
            f"tool{DUMMIES_SEPARATOR}toaster",
            f"score{DUMMIES_SEPARATOR}aa",
            f"score{DUMMIES_SEPARATOR}ab",
        ],
    )


def test_transform(mixed_df2: pd.DataFrame) -> None:
    df = mixed_df2

    _, model = fit_transform(df)

    df_transformed = transform(df, model)
    df_expected = pd.DataFrame(
        {
            "Dim. 1": [2.0, -2],
            "Dim. 2": [0.0, 0],
        }
    )

    assert_frame_equal(df_transformed.abs(), df_expected.abs())


def test_transform_vs_coord(mixed_df2: pd.DataFrame) -> None:
    df = mixed_df2

    coord, model = fit_transform(df)
    df_transformed = transform(df, model)

    assert_frame_equal(df_transformed, coord)


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )
    result, _ = fit_transform(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [0.0, 0.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


def test_scaler_pca_famd(mixed_df2: pd.DataFrame) -> None:
    original_df = mixed_df2

    _, model = fit_transform(original_df)
    df = scaler(model, original_df)

    _, model_pca = fit_pca(original_df[model.original_continuous])
    df_pca = scaler_pca(model_pca, original_df)

    assert_frame_equal(df[model.original_continuous], df_pca[model.original_continuous])


def test_center_pca_famd(mixed_df2: pd.DataFrame) -> None:
    original_df = mixed_df2

    _, model = fit_transform(original_df)
    continous = model.original_continuous
    categorical = model.original_categorical
    df, mean1, std1, _, _ = center(original_df, quali=categorical, quanti=continous)

    df_pca, mean2, std2 = center_pca(original_df[continous])

    assert_frame_equal(df[continous], df_pca[continous])

    assert_series_equal(mean1, mean2)
    assert_series_equal(std1, std2)


def test_get_variable_contributions_same_as_factomineR(iris_df: pd.DataFrame) -> None:
    """Verify that we get the same contributions and cos2 with factormineR."""
    df = iris_df
    _, model = fit_transform(df, nf=5)

    contributions, cos2 = get_variable_contributions(model, df, explode=True)

    contributions_path = Path("tests/fixtures/iris_factomineR_contributions.csv")
    cos2_path = Path("tests/fixtures/iris_factomineR_cos2.csv")

    expected_contributions = pd.read_csv(contributions_path)
    expected_cos2 = pd.read_csv(cos2_path)

    np.testing.assert_array_almost_equal(contributions, expected_contributions)
    np.testing.assert_array_almost_equal(cos2, expected_cos2, decimal=4)


def test_get_variable_contributions(mixed_df: pd.DataFrame) -> None:
    """Verify that the contributions and cos2 are the ones expected."""
    df = mixed_df
    _, model = fit_transform(df, nf=3)

    contributions, cos2 = get_variable_contributions(model, df, explode=False)

    expected_contributions = pd.DataFrame.from_dict(
        data={
            "variable_1": [50, 50, 0],
            "tool": [50, 50, 100],
        },
        dtype=np.float64,
        orient="index",
        columns=get_projected_column_names(3),
    )

    expected_cos2 = pd.DataFrame.from_dict(
        data={
            "variable_1": [0.897214, 0.002786, 0],
            "tool": [0.897214, 0.002786, 0.25],
        },
        orient="index",
        columns=get_projected_column_names(3),
    )

    assert_frame_equal(
        contributions, expected_contributions, check_exact=False, atol=0.0001
    )
    assert_frame_equal(cos2, expected_cos2, check_exact=False, atol=0.0001)


@pytest.mark.parametrize("col_weights", [[2.0, 3.0], None])
def test_get_variable_contributions_exploded_parameter(
    mixed_df: pd.DataFrame, col_weights: Any
) -> None:
    """Verify argument explode=False and explode=True in get_variable_contributions.

    Make sure that explode=False is the sum of explode=True for categorical variables.
    Also make sure that this remains true with any col_weights.
    """
    df = mixed_df
    variable = "tool"
    _, model = fit_transform(df, nf=3, col_weights=col_weights)

    contributions_exploded, _ = get_variable_contributions(model, df, explode=True)
    contributions_not_exploded, _ = get_variable_contributions(model, df, explode=False)

    dummies = filter(
        lambda name: f"{variable}{DUMMIES_SEPARATOR}" in name,
        contributions_exploded.index,
    )
    assert_series_equal(
        contributions_exploded.loc[list(dummies)].sum(),
        contributions_not_exploded.loc[variable],
        check_names=False,
    )


def test_get_variable_contributions_with_multiple_variables(
    quali_df: pd.DataFrame, quanti_df: pd.DataFrame
) -> None:
    """Verify that contributions can be computed using multiple categorical columns.

    This is a regression test.
    """
    # We have to combine quali_df and quanti_df (and not use mixed_df or mixed_df_2)
    # because the latter have perfectly correlated columns which yield a division by zero.
    df = pd.concat([quali_df, quanti_df], axis="columns")
    _, model = fit_transform(df, nf=4)
    get_variable_contributions(model, df, explode=True)


def test_get_variable_contributions_sum_is_100_with_col_weights_random_famd(
    mixed_df: pd.DataFrame,
) -> None:
    model = fit(mixed_df, col_weights=[3.0, 2.0])  # type: ignore
    contributions, _ = get_variable_contributions(model, mixed_df)
    summed_contributions = contributions.sum(axis=0)
    assert_series_equal(summed_contributions, pd.Series([100.0] * 3), check_index=False)


def test_get_variable_contributions_with_constant_variable() -> None:
    """Ensure it handles a constant variable in the df, hence with a null eigenvalue."""
    df = pd.DataFrame(
        {
            "quantitative_var": [1, 2, 3, 4],
            "quantitative_constant_var": [1, 1, 1, 1],
            "qualitative_var": ["a", "b", "c", "d"],
            "qualitative_constant_var": ["a", "a", "a", "a"],
        }
    )
    _, model = fit_transform(df, nf=None)

    contributions, _ = get_variable_contributions(model, df, explode=False)

    assert np.isfinite(contributions).all().all()


def test_reconstructed_df_from_model_equals_df_minimal(mixed_df: pd.DataFrame) -> None:
    """Ensure that the reconstructed df from the model is equal to the original df."""
    df = mixed_df
    model = fit(df)
    reconstructed_df = reconstruct_df_from_model(model)
    # don't check dtypes, model don't know if numerical were int or float
    assert_frame_equal(df, reconstructed_df, check_dtype=False)


def test_reconstructed_df_from_weighted_model_equals_df() -> None:
    """Ensure that the reconstructed df from the model is equal to the original df."""
    df = pd.read_csv("./fixtures/iris.csv")
    model = fit(df, col_weights=[3, 1, 1, 1, 1])  # type: ignore
    reconstructed_df = reconstruct_df_from_model(model)
    assert_frame_equal(df, reconstructed_df)
