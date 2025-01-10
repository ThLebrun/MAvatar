from typing import cast

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from saiph.reduction import DUMMIES_SEPARATOR
from saiph.reduction.famd_sparse import (
    center_sparse,
    fit,
    fit_transform,
    scaler_sparse,
    transform,
)
from saiph.reduction.pca import center as center_pca
from saiph.reduction.pca import fit_transform as fit_pca
from saiph.reduction.pca import scaler as scaler_pca


def test_fit_mix(mixed_df2: pd.DataFrame) -> None:
    df = mixed_df2.iloc[:, 0:3]

    result, model = fit_transform(df)
    model.mean = cast(pd.Series, model.mean)
    model.std = cast(pd.Series, model.std)
    model.prop = cast(pd.Series, model.prop)
    model.s = cast(NDArray[np.float64], model.s)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [-1.73, 1.73],
            "Dim. 2": [1.41, 1.41],
        }
    )
    expected_v: NDArray[np.float64] = np.array(
        [
            [-5.773503e-01, -4.082483e-01, 4.082483e-01, 4.082483e-01, -4.082483e-01],
            [5.384773e-16, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01],
        ]
    )
    expected_s: NDArray[np.float64] = np.array([1.224745e00, 1])
    expected_u: NDArray[np.float64] = np.array([[1.0, 1.0], [-1.0, 1.0]])
    expected_explained_var: NDArray[np.float64] = np.array([1.5, 1.0])
    expected_explained_var_ratio: NDArray[np.float64] = np.array([0.6, 0.4])

    assert_frame_equal(abs(result), abs(expected_result), check_exact=False, atol=0.01)
    assert_allclose(np.abs(model.V), np.abs(expected_v), atol=0.01)
    assert_allclose(model.s, expected_s, atol=0.01)
    assert_allclose(np.abs(model.U), np.abs(expected_u), atol=0.01)
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
    assert model.D_c is None

    assert_allclose(model.mean, np.array(2.5))
    assert_allclose(model.std, np.array(1.5))

    assert_allclose(
        model.prop,
        [0.5, 0.5, 0.5, 0.5],
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

    model = fit(df)

    df_transformed = transform(df, model)
    df_expected = pd.DataFrame(
        {
            "Dim. 1": [2.0, -2],
            "Dim. 2": [1.414214, 1.414214],
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
            "Dim. 1": [1.414213562373095, 1.414213562373095],
            "Dim. 2": [-1.110223e-16, -1.110223e-16],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


def test_scaler_pca_famd(mixed_df2: pd.DataFrame) -> None:
    """Compare FAMD and PCA numeric scaler results."""
    original_df = mixed_df2

    _, model = fit_transform(original_df)
    df_famd = scaler_sparse(model, original_df)

    _, model_pca = fit_pca(original_df[model.original_continuous])
    df_pca = scaler_pca(model_pca, original_df)

    assert_array_equal(
        df_famd.todense()[:, [0, 1]], df_pca[model.original_continuous].to_numpy()
    )


def test_center_pca_famd(mixed_df2: pd.DataFrame) -> None:
    """Compare FAMD and PCA numeric center results."""
    original_df = mixed_df2

    _, model = fit_transform(original_df)
    continuous = model.original_continuous
    categorical = model.original_categorical
    df_famd, mean1, std1, _, _ = center_sparse(
        original_df, quali=categorical, quanti=continuous
    )

    df_pca, mean2, std2 = center_pca(original_df[continuous])

    assert_array_equal(df_famd.todense()[:, [0, 1]], df_pca.to_numpy())

    assert_series_equal(mean1, mean2)
    assert_series_equal(std1, std2)
