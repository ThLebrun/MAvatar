from typing import cast

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pandas.testing import assert_frame_equal

from saiph.reduction.pca import (
    center,
    fit,
    fit_transform,
    reconstruct_df_from_model,
    scaler,
    transform,
)


def test_fit_scale() -> None:
    df = pd.DataFrame(
        {
            "one": [1.0, 3.0],
            "two": [2.0, 4.0],
        }
    )

    result, model = fit_transform(df)
    model.mean = cast(pd.Series, model.mean)
    model.std = cast(pd.Series, model.std)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [-1.41, 1.41],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v: NDArray[np.float64] = np.array([[0.71, 0.71], [-0.71, 0.71]])
    expected_explained_var: NDArray[np.float64] = np.array([1.0, 0.0])
    expected_explained_var_ratio: NDArray[np.float64] = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_allclose(np.absolute(model.V), np.absolute(expected_v), atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [2.0, 3.0])
    assert_allclose(model.std, [1.0, 1.0])


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "one": [1, 1],
            "two": [2, 2],
        }
    )

    result, model = fit_transform(df)
    model.mean = cast(pd.Series, model.mean)
    model.std = cast(pd.Series, model.std)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.0, 0.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v: NDArray[np.float64] = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected_explained_var: NDArray[np.float64] = np.array([0.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    # np.nan == np.nan returns False
    assert pd.isnull(model.explained_var_ratio).all()
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [1.0, 2.0])
    assert_allclose(
        model.std, [0, 0]
    )  # The standard deviation of a constant variable is zero


def test_center_scaler() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 12.0],
            1: [2.0, 4.0],
        }
    )

    _, model = fit_transform(df)

    df1, _, _ = center(df)
    df2 = scaler(model, df)

    assert_frame_equal(df1, df2)


def test_transform_simple() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 12.0],
            1: [2.0, 4.0],
        }
    )

    _, model = fit_transform(df)

    df_transformed = transform(df, model)

    expected_transformed = pd.DataFrame(
        {
            "Dim. 1": [-1.414214, 1.414214],
            "Dim. 2": [0.0, 0.0],
        }
    )

    assert_frame_equal(df_transformed, expected_transformed, atol=0.01)


def test_transform() -> None:
    df = pd.DataFrame({0: [-2.0, 7.0, -4.5], 1: [6.0, 2.0, 7.0], 2: [5.0, 10.0, -14.5]})

    _, model = fit_transform(df)

    df_transformed = transform(df, model)

    expected_transformed = pd.DataFrame(
        {
            "Dim. 1": [-0.285647, 2.150812, -1.865165],
            "Dim. 2": [0.730941, -0.287485, -0.443456],
            "Dim. 3": [0.0, 0.0, 0.0],
        }
    )

    assert_frame_equal(
        df_transformed, expected_transformed, check_exact=False, atol=0.00001
    )


def test_transform_vs_coord() -> None:
    df = pd.DataFrame({0: [-2.0, 7.0, -4.5], 1: [6.0, 2.0, 7.0], 2: [5.0, 10.0, -14.5]})
    coord, model = fit_transform(df)
    df_transformed = transform(df, model)

    assert_frame_equal(coord, df_transformed)


def test_reconstructed_df_from_model_equals_df_minimal() -> None:
    """Ensure that the reconstructed df from the model is equal to the original df."""
    df = pd.DataFrame({0: [-2.0, 7.0, -4.5], 1: [6.0, 2.0, 7.0], 2: [5.0, 10.0, -14.5]})
    model = fit(df)
    reconstructed_df = reconstruct_df_from_model(model)
    # don't check dtypes, model don't know if numerical were int or float
    assert_frame_equal(df, reconstructed_df)


def test_reconstructed_df_from_weighted_model_equals_df() -> None:
    """Ensure that the reconstructed df from the model is equal to the original df."""
    df = pd.read_csv("./fixtures/wbcd.csv").astype(int)
    model = fit(df, col_weights=[3, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # type: ignore
    reconstructed_df = reconstruct_df_from_model(model)
    assert_frame_equal(df, reconstructed_df)
