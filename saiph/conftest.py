import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from saiph.models import Model
from saiph.reduction import DUMMIES_SEPARATOR

_iris_csv = pd.read_csv("tests/fixtures/iris.csv")
_wbcd_csv = pd.read_csv("tests/fixtures/breast_cancer_wisconsin.csv")
_wbcd_supplemental_coordinates_csv_mca = pd.read_csv(
    "tests/fixtures/wbcd_supplemental_coordinates_mca.csv"
)
_wbcd_supplemental_coordinates_csv_pca = pd.read_csv(
    "tests/fixtures/wbcd_supplemental_coordinates_pca.csv"
)
_wbcd_supplemental_coordinates_csv_famd = pd.read_csv(
    "tests/fixtures/wbcd_supplemental_coordinates_famd.csv"
)


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return _iris_csv.copy()


@pytest.fixture
def iris_quanti_df() -> pd.DataFrame:
    return _iris_csv.drop("variety", axis=1).copy()


@pytest.fixture()
def quanti_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7],
            "variable_2": [10, 20, 30, 40],
            "variable_3": [100, 50, -30, -50],
        }
    )


@pytest.fixture()
def quali_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tool": ["wrench", "wrench", "hammer", "hammer"],
            "fruit": ["apple", "orange", "apple", "apple"],
        }
    )


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7],
            "tool": ["wrench", "wrench", "hammer", "hammer"],
        }
    )


@pytest.fixture
def mixed_df2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
            "age": [55, 62],
        }
    )


@pytest.fixture
def wbcd_quali_df() -> pd.DataFrame:
    """Wisconsin breast cancer dataframe.

    Columns are categorical variables.
    """
    return _wbcd_csv.drop(columns=["Sample_code_number"]).astype("category").copy()


@pytest.fixture
def wbcd_quanti_df() -> pd.DataFrame:
    """Wisconsin breast cancer dataframe.

    Columns are continuous variables.
    """
    return _wbcd_csv.drop(columns=["Sample_code_number"]).astype("int").copy()


@pytest.fixture
def wbcd_mixed_df() -> pd.DataFrame:
    """Wisconsin breast cancer dataframe.

    Columns are mixed of continuous and categorical variables.
    """
    wbcd_mixed = _wbcd_csv.drop(columns=["Sample_code_number"]).astype("int").copy()
    wbcd_mixed[["Class", "Mitoses", "Normal_Nucleoli", "Bland_Chromatin"]] = wbcd_mixed[
        ["Class", "Mitoses", "Normal_Nucleoli", "Bland_Chromatin"]
    ].astype("category")
    return wbcd_mixed


@pytest.fixture
def wbcd_supplemental_coord_quali() -> pd.DataFrame:
    """Supplemental coordinates of the WBCD dataset, generated using MCA model."""
    return _wbcd_supplemental_coordinates_csv_mca.copy()


@pytest.fixture
def wbcd_supplemental_coord_quanti() -> pd.DataFrame:
    """Supplemental coordinates of the WBCD dataset, generated using PCA model."""
    return _wbcd_supplemental_coordinates_csv_pca.copy()


@pytest.fixture
def wbcd_supplemental_coord_mixed() -> pd.DataFrame:
    """Supplemental coordinates of the WBCD dataset, generated using FAMD model.

    Categorical columns are Class, Mitoses, Normal_Nucleoli, Bland_Chromatin.
    """
    return _wbcd_supplemental_coordinates_csv_famd.copy()


@pytest.fixture
def mapping() -> Dict[str, List[str]]:
    sep = DUMMIES_SEPARATOR
    return {
        "tool": [f"tool{sep}hammer", f"tool{sep}wrench"],
        "fruit": [f"fruit{sep}apple", f"fruit{sep}orange"],
    }


def check_model_equality(
    test: Model,
    expected: Model,
) -> None:
    """Verify that two Model instances are the same."""
    for key, value in expected.__dict__.items():
        test_item = test.__dict__[key]
        expected_item = value
        check_equality(test_item, expected_item)


def check_equality(
    test: Any,
    expected: Any,
) -> None:
    """Check equality of dataframes, series and np.arrays."""
    with warnings.catch_warnings():
        # We ignore the warning about mismatched null-like values
        # In a future update of pandas, this will raise when comparing
        # None and np.nan (which is what we want)
        warnings.filterwarnings(
            action="ignore",
            message="Mismatched null-like values None and nan found.",
            category=FutureWarning,
        )
        if isinstance(test, pd.DataFrame) and isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(test, expected)
        elif isinstance(test, pd.Series) and isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(test, expected)
        elif isinstance(test, np.ndarray) and isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(test, expected)
        else:
            assert test == expected
