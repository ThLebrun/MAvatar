from contextlib import nullcontext
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pytest
from doubles import expect
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

import saiph
from saiph import projection
from saiph.exception import ColumnsNotFoundError, InvalidParameterException
from saiph.inverse_transform import inverse_transform
from saiph.projection import (
    fit,
    fit_transform,
    get_variable_contributions,
    stats,
    transform,
)
from saiph.reduction.utils.common import get_projected_column_names


def test_transform_then_inverse_FAMD(iris_df: pd.DataFrame) -> None:
    transformed, model = fit_transform(iris_df, nf=None)
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, iris_df)


def test_transform_then_inverse_PCA(iris_quanti_df: pd.DataFrame) -> None:
    transformed, model = fit_transform(iris_quanti_df, nf=None)
    un_transformed = inverse_transform(transformed, model)
    assert_frame_equal(un_transformed, iris_quanti_df)


def test_transform_then_inverse_MCA(quali_df: pd.DataFrame) -> None:
    df = quali_df
    transformed, model = fit_transform(df)
    un_transformed = inverse_transform(transformed, model)
    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_MCA_type(quali_df: pd.DataFrame) -> None:
    df = quali_df

    df = df.astype("object")
    transformed, model = fit_transform(df)
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_FAMD_weighted(mixed_df: pd.DataFrame) -> None:
    df = mixed_df
    col_weights: Dict[str, Union[int, float]] = {"variable_1": 2, "tool": 3}
    transformed, model = fit_transform(df, col_weights=col_weights)
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_PCA_weighted(quanti_df: pd.DataFrame) -> None:
    df = quanti_df
    col_weights: Dict[str, Union[int, float]] = {
        "variable_1": 2,
        "variable_2": 1,
        "variable_3": 3,
    }
    coords, model = fit_transform(df, col_weights=col_weights)
    un_transformed = inverse_transform(coords, model)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_MCA_weighted() -> None:
    df = pd.DataFrame(
        {
            "variable_1": ["1", "3", "3", "3", "1", "2", "2", "1", "1", "2"],
            "variable_2": ["1", "1", "1", "2", "2", "1", "1", "1", "2", "2"],
            "variable_3": ["1", "2", "1", "2", "1", "2", "1", "1", "2", "2"],
            "variable_4": [
                "red",
                "blue",
                "blue",
                "green",
                "red",
                "blue",
                "red",
                "red",
                "red",
                "red",
            ],
        }
    )
    col_weights: Dict[str, Union[int, float]] = {
        "variable_1": 2,
        "variable_2": 1,
        "variable_3": 3,
        "variable_4": 2,
    }
    transformed, model = fit_transform(df, col_weights=col_weights)
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, df)


df_pca = pd.DataFrame(
    {
        0: [
            1000.0,
            3000.0,
            10000.0,
            1500.0,
            700.0,
            3300.0,
            5000.0,
            2000.0,
            1200.0,
            6000.0,
        ],
        1: [185.0, 174.3, 156.8, 182.7, 180.3, 179.2, 164.7, 192.5, 191.0, 169.2],
        2: [1, 5, 10, 2, 4, 4, 7, 3, 1, 6],
    }
)

df_famd = pd.DataFrame(
    {
        "variable_1": [4, 5, 6, 7, 11, 2, 52],
        "variable_2": [10, 20, 30, 40, 10, 74, 10],
        "variable_3": ["red", "blue", "blue", "green", "red", "blue", "red"],
        "variable_4": [100, 50, -30, -50, -19, -29, -20],
    }
)

df_mca = pd.DataFrame(
    {
        0: [
            "red",
            "red",
            "whithe",
            "whithe",
            "red",
            "whithe",
            "red",
            "red",
            "whithe",
            "red",
        ],
        1: [
            "beef",
            "chicken",
            "fish",
            "fish",
            "beef",
            "chicken",
            "beef",
            "chicken",
            "fish",
            "beef",
        ],
        2: [
            "france",
            "espagne",
            "france",
            "italy",
            "espagne",
            "france",
            "france",
            "espagne",
            "chine",
            "france",
        ],
    }
)


@pytest.mark.parametrize(
    "df_input,expected_type", [(df_pca, "pca"), (df_mca, "mca"), (df_famd, "famd")]
)
def test_eval(df_input: pd.DataFrame, expected_type: str) -> None:
    _, model = fit_transform(df_input)
    assert model.type == expected_type


# check cor of variables to dim (if cor is ok so is cos2)

expected_pca_cor = [0.9621683005738202, -0.9667100394109722, 0.9803843201246043]
expected_mca_cor = [
    -0.9169410964961192,
    0.9169410964961192,
    -0.5776131247092218,
    -0.3215176498417082,
    0.9390120540605189,
    0.5586386162833192,
    -0.5628216589197227,
    -0.15453176858793358,
    0.5586386162833191,
]
expected_famd_cor = [
    -0.5930925452224494,
    0.8992562362632682,
    -0.5058995881660323,
    0.6216213705726737,
    0.399494477072544,
    -0.9041066243572436,
]


@pytest.mark.parametrize(
    "df_input,expected_cor",
    [
        (df_pca, expected_pca_cor),
        (df_mca, expected_mca_cor),
        (df_famd, expected_famd_cor),
    ],
)
def test_var_cor(df_input: pd.DataFrame, expected_cor: List[float]) -> None:
    _, model = fit_transform(df_input)
    stats(model, df_input)
    if model.correlations is not None:
        assert_allclose(model.correlations["Dim. 1"], expected_cor, atol=1e-07)


# Check percentage of explained variance

expected_pca_explained_var_ratio = [
    0.9404831846910865,
    0.040151017139633684,
    0.01936579816927985,
]
expected_mca_explained_var_ratio = [
    0.42099362789799255,
    0.23253662367291794,
    0.1666666666666665,
    0.1314458557658021,
    0.035220810900864555,
]
expected_famd_explained_var_ratio = [
    0.44820992177856206,
    0.2801754650846534,
    0.1707856006031922,
    0.05566366403780453,
    0.04516534849578783,
]


@pytest.mark.parametrize(
    "df_input,expected_var_ratio",
    [
        (df_pca, expected_pca_explained_var_ratio),
        (df_mca, expected_mca_explained_var_ratio),
        (df_famd, expected_famd_explained_var_ratio),
    ],
)
def test_var_ratio(df_input: pd.DataFrame, expected_var_ratio: List[float]) -> None:
    _, model = fit_transform(df_input)
    stats(model, df_input)
    assert_allclose(model.explained_var_ratio[0:5], expected_var_ratio, atol=1e-07)


# FutureWarning: In a future version, the Index constructor will not infer numeric
# dtypes when passed object-dtype sequences (matching Series behavior)
@pytest.mark.filterwarnings(
    "ignore:In a future version, the Index constructor will not infer numeric dtypes"
)
@pytest.mark.parametrize("dtypes", ["object", "category"])
# This test checks that inverse_transform o transform = id in those cases:
# - We have a categorical variable of BOOLEANS True | False
# - We have a categorical variable of STRINGS "True" | "False"
def test_transform_then_inverse_value_type(dtypes: str) -> None:
    """Test the type of the value of each variable."""
    df = pd.DataFrame(
        {
            "variable_1": [1, 1, 2, 2, 1, 1, 2, 1],
            "variable_2": ["1", "2", "2", "2", "1", "2", "1", "2"],
            "variable_3": [True, True, False, True, True, False, True, False],
            "variable_4": [
                "True",
                "True",
                "False",
                "True",
                "True",
                "False",
                "True",
                "False",
            ],
        }
    )
    df = df.astype(dtypes)

    coord, model = fit_transform(df)
    result = inverse_transform(coord, model)

    assert_frame_equal(df, result)


def test_get_variable_contribution_for_pca(quanti_df: pd.DataFrame) -> None:
    """Verify that get_variable_contributions returns the corrections contributions for PCA.

    FIXME:There is no unique function in reduction/pca.py that computes the contributions
    because it is a very simple task. However, it might be useful to refactor as it
    will improve readability.
    """
    df = quanti_df
    _, model = fit_transform(df, nf=3)

    contributions = get_variable_contributions(model, df)
    expected_contributions = pd.DataFrame.from_dict(
        data={
            "variable_1": [33.497034, 16.502966, 33.465380],
            "variable_2": [33.497034, 16.502966, 33.465380],
            "variable_3": [33.005932, 66.994068, 33.069241],
        },
        dtype=np.float64,
        orient="index",
        columns=get_projected_column_names(3),
    )

    # We only look at n - 1 first dimensions because of roundoff errors.
    # https://github.com/octopize/saiph/issues/58
    assert_frame_equal(contributions.iloc[:, :-1], expected_contributions.iloc[:, :-1])


def test_get_variable_contribution_are_similar_with_reduced_nf() -> None:
    """Verify that get_variable_contributions returns similar contributions with reduced nf."""
    df = pd.read_csv("./fixtures/wbcd.csv")
    model_truncated = fit(df, nf=5)  # nf = nf_max/2
    df_reconstructed = projection.get_reconstructed_df_from_model(model_truncated)
    truncated_contributions = get_variable_contributions(
        model_truncated, df_reconstructed
    )
    model_full = fit(df)  # nf = nf_max
    full_contributions = get_variable_contributions(model_full, df)
    # We only look at the first two dimensions
    truncated_contributions = truncated_contributions.iloc[:, :2]
    full_contributions = full_contributions.iloc[:, :2]
    # Filter values above 1 in both DataFrames
    mask = (truncated_contributions > 1) & (full_contributions > 1)
    truncated_contributions = truncated_contributions[mask]
    full_contributions = full_contributions[mask]

    # Calculate absolute differences
    absolute_differences = (truncated_contributions - full_contributions).abs()

    # Sum of absolute differences
    total_difference = absolute_differences.sum().sum()

    # Sum of all elements in the full_nf DataFrame
    total_sum = full_contributions.sum().sum()

    # Calculate the percentage difference
    percentage_difference = (total_difference / total_sum) * 100

    assert percentage_difference < 20


def test_get_variable_contributions_calls_correct_subfunction(
    quanti_df: pd.DataFrame, quali_df: pd.DataFrame, mixed_df: pd.DataFrame
) -> None:
    """Verify that projection.get_variable_contributions calls the correct subfunction."""
    # FAMD
    model = fit(mixed_df)
    expect(saiph.reduction.famd).get_variable_contributions(
        model, mixed_df, explode=False
    ).once().and_return((None, None))
    projection.get_variable_contributions(model, mixed_df)

    # MCA
    model = fit(quali_df)
    expect(saiph.reduction.mca).get_variable_contributions(
        model, quali_df, explode=False
    ).once().and_return(None)
    projection.get_variable_contributions(model, quali_df)

    # PCA
    model = fit(quanti_df)
    expect(saiph.projection).get_variable_correlation(
        model, quanti_df
    ).once().and_return(pd.DataFrame([1, 2, 3]))
    projection.get_variable_contributions(model, quanti_df)


def test_stats_calls_correct_subfunction(
    quali_df: pd.DataFrame, mixed_df: pd.DataFrame
) -> None:
    """Verify that projection.stats calls the correct subfunction."""
    # FAMD
    model = fit(mixed_df)
    expect(saiph.reduction.famd).stats(
        model, mixed_df, explode=False
    ).once().and_return(model)

    # MCA
    projection.stats(model, mixed_df)
    model = fit(quali_df)
    expect(saiph.reduction.mca).stats(model, quali_df, explode=False).once().and_return(
        model
    )
    projection.stats(model, quali_df)

    # FIXME: Can't test PCA as it has no subfunction associated to it


@pytest.mark.parametrize(
    "nf,expectation",
    [
        # Invalid boundary
        (0, pytest.raises(InvalidParameterException)),
        (5, pytest.raises(InvalidParameterException)),
        # Valid boundary
        (4, nullcontext()),
        (1, nullcontext()),
        (None, nullcontext()),
    ],
)
def test_fit_checks_nf_parameter(
    quali_df: pd.DataFrame,
    nf: int,
    expectation: Any,
) -> None:
    """Verify that fit checks nf parameter and fails when it needs to."""
    with expectation:
        fit(quali_df, nf=nf)


@pytest.mark.parametrize(
    "col_weights,expectation",
    [
        # Invalid
        ({"col": 2}, pytest.raises(InvalidParameterException)),
        # Valid
        ({"tool": 2, "fruit": 3}, nullcontext()),
        ({"fruit": 3}, nullcontext()),
        (None, nullcontext()),
    ],
)
def test_fit_checks_col_weights_parameter(
    quali_df: pd.DataFrame,
    col_weights: Any,
    expectation: Any,
) -> None:
    """Verify that fit checks col_weights parameter and fails when it needs to."""
    with expectation:
        fit(quali_df, col_weights=col_weights)


@pytest.mark.parametrize(
    "starting_seed, stored_seed",
    [
        (1, 2032329982),
        (np.random.default_rng(1), 2032329982),  # same stored_seed as seed=1
        (2, 3597359296),  # different than seed=1
    ],
)
def test_fit_pca_works_with_different_arguments_for_seed_and_stores_them_in_model(
    starting_seed: Any, stored_seed: int
) -> None:
    """Verify that fit works with different arguments for seed and stores them in model."""
    model = fit(df_pca, seed=starting_seed)
    assert model.seed == stored_seed


@pytest.mark.parametrize(
    "starting_seed, stored_seed",
    [
        (1, 2032329982),
        (np.random.default_rng(1), 2032329982),  # same stored_seed as seed=1
        (2, 3597359296),  # different than seed=1
    ],
)
def test_fit_famd_works_with_different_arguments_for_seed_and_stores_them_in_model(
    starting_seed: Any, stored_seed: int
) -> None:
    """Verify that fit works with different arguments for seed and stores them in model."""
    model = fit(df_famd, seed=starting_seed)
    assert model.seed == stored_seed


@pytest.mark.parametrize(
    "starting_seed, stored_seed",
    [
        (1, 2032329982),
        (np.random.default_rng(1), 2032329982),  # same stored_seed as seed=1
        (2, 3597359296),  # different than seed=1
    ],
)
def test_fit_mca_works_with_different_arguments_for_seed_and_stores_them_in_model(
    starting_seed: Any, stored_seed: int
) -> None:
    """Verify that fit works with different arguments for seed and stores them in model."""
    model = fit(df_mca, seed=starting_seed)
    assert model.seed == stored_seed


def test_get_reconstructed_df_from_model_calls_correct_subfunction(
    quanti_df: pd.DataFrame, quali_df: pd.DataFrame, mixed_df: pd.DataFrame
) -> None:
    """Verify that projection.get_reconstructed_df_from_model calls the correct subfunction."""
    # FAMD
    model = fit(mixed_df)
    expect(saiph.reduction.famd).reconstruct_df_from_model(model).once().and_return(
        (None)
    )
    projection.get_reconstructed_df_from_model(model)

    # MCA
    model = fit(quali_df)
    expect(saiph.reduction.mca).reconstruct_df_from_model(model).once().and_return(
        (None)
    )
    projection.get_reconstructed_df_from_model(model)

    # PCA
    model = fit(quanti_df)
    expect(saiph.reduction.pca).reconstruct_df_from_model(model).once().and_return(
        (None)
    )
    projection.get_reconstructed_df_from_model(model)


@pytest.mark.parametrize(
    "df_to_fit,df_to_transform",
    [
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        ),
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        ),
    ],
)
def test_transform_raise_error_on_wrong_columns(
    df_to_fit: pd.DataFrame, df_to_transform: pd.DataFrame
) -> None:
    """Verify that transform raises error when columns in the model and the df are different."""
    model = fit(df_to_fit)
    with pytest.raises(ColumnsNotFoundError):
        transform(df_to_transform, model)


@pytest.mark.parametrize(
    "df_to_fit_transform",
    [
        pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
        ),  # pca
        pd.DataFrame(
            {"a": ["a", "b", "b"], "b": [1, 3, 6], "c": [1, 2, 3], "d": [1, 2, 3]}
        ),  # famd
        pd.DataFrame(
            {
                "a": ["a", "b", "b"],
                "b": ["a", "b", "b"],
                "c": ["a", "b", "b"],
                "d": ["a", "b", "b"],
            }
        ),  # mca
    ],
)
def test_fit_transform_with_horizontal_matrix(
    df_to_fit_transform: pd.DataFrame,
) -> None:
    """Verify that the coordinates are a squared matrix even if the input is horizontal."""
    coord, __ = fit_transform(df_to_fit_transform)
    assert coord.shape[0] == coord.shape[1]
