import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics.pairwise import euclidean_distances

from gower import gower_matrix

from utils.Generators.CTGAN import CTGAN_generator
from utils.Generators.Kanonymiser import Kanon
from utils.Generators.Identity import Identity
from utils.Generators.MST import MST
from utils.Generators.SAIPH_Compressor import SAIPH_Compressor
from utils.Generators.MAvatar import MAvatar

from itertools import zip_longest
from copy import deepcopy

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


def pp_prepare(X_ori, transformer=None):
    X = X_ori.copy()
    # Identify numeric and categorical columns
    X.reset_index(inplace=True)
    if "index" in X.columns:
        X = X.drop(columns=["index"])
    if transformer is None:
        transformer = {
            "numeric": None,
            "categorical": None,
            "original_columns": list(X.columns),
            "preprocessed_cols": None,
            "type_col": {
                col_name: X.dtypes[col_name] for col_name in list(X.columns)
            },
        }
    return X, transformer


def pp_numerical(X, scaler, transfo=None):
    transfo["numeric_cols"] = X.select_dtypes(
        include=["int64", "float64"]
    ).columns
    if len(transfo["numeric_cols"]) > 0:
        if transfo["numeric"] is None:
            X[transfo["numeric_cols"]] = scaler.fit_transform(
                X[transfo["numeric_cols"]]
            )
            transfo["numeric"] = scaler
        else:
            scaler = transfo["numeric"]
            X[transfo["numeric_cols"]] = scaler.transform(
                X[transfo["numeric_cols"]]
            )
    return X, transfo


def post_numerical(X, transfo):
    numeric_columns = transfo["numeric_cols"]
    print(numeric_columns)
    if len(numeric_columns) > 0:
        scaler = transfo["numeric"]
        X[numeric_columns] = scaler.inverse_transform(X[numeric_columns])
    for num_col in numeric_columns:
        X[num_col] = X[num_col].astype(transfo["type_col"][num_col])
    return X


def pp_OneHot(X, transformer):
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns
    if len(categorical_columns) > 0:
        if transformer["categorical"] is None:
            encoder = OneHotEncoder(drop="first", sparse_output=False)
            X_encoded = encoder.fit_transform(X[categorical_columns])
            transformer["categorical"] = encoder
            transformer["categorical_columns"] = categorical_columns
        else:
            encoder = transformer["categorical"]
            X_encoded = encoder.transform(
                X[transformer["categorical_columns"]]
            )

        # Get the column names after one-hot encoding
        encoded_columns = encoder.get_feature_names_out(
            transformer["categorical_columns"]
        )
        transformer["encoded_columns"] = encoded_columns
        # Create a mapping of new column names
        column_maps = list(transformer["numeric_cols"]) + list(encoded_columns)

        # Concatenate the dataframes
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
        X = pd.concat(
            [X.drop(categorical_columns, axis=1), X_encoded_df],
            axis=1,
            ignore_index=True,
        )
        old_cols = list(X.columns)
        column_mapping = {
            old_cols[i]: new_name for i, new_name in enumerate(column_maps)
        }
        X = X.rename(columns=column_mapping)
    else:
        transformer["categorical"] = None
        transformer["categorical_columns"] = list()
        transformer["encoded_columns"] = list()
    return X, transformer


def post_categorical(X, transformer):
    if not (transformer["categorical"] is None):
        encoder = transformer["categorical"]

        X_encoded = X[transformer["encoded_columns"]]
        # Inverse transform the one-hot encoded data
        X_decoded = pd.DataFrame(
            encoder.inverse_transform(X_encoded),
            columns=transformer["categorical_columns"],
        )

        # Replace the original categorical columns with the decoded ones
        X[transformer["categorical_columns"]] = X_decoded
        # Drop the one-hot encoded columns
        X = X.drop(transformer["encoded_columns"], axis=1)
    return X


def pp_Adjust(X, transfo):
    if transfo["preprocessed_cols"] is not None:
        for col in transfo["preprocessed_cols"]:
            if col not in list(X.columns):
                X[col] = np.zeros(len(X))
        X_temp = deepcopy(X)
        X = X_temp[transfo["preprocessed_cols"]]
    else:
        transfo["preprocessed_cols"] = X.columns
    # Ensure the column names are strings
    X.columns = X.columns.astype(str)
    return X, transfo


def preprocess(X_ori, scaler_name="Standard", transformer=None):
    scaler_dict = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(feature_range=(-1, 1)),
    }
    X, new_transformer = pp_prepare(X_ori, transformer)
    # print(scaler_dict)
    X, new_transformer = pp_numerical(
        X, scaler_dict[scaler_name], new_transformer
    )
    X, new_transformer = pp_OneHot(X, new_transformer)
    X, new_transformer = pp_Adjust(X, new_transformer)
    return X, new_transformer


def postprocess(X_ori, transformer):
    X = post_numerical(X_ori.copy(), transformer)
    X = post_categorical(X, transformer)
    return X[transformer["original_columns"]]


def preprocess_double(X_train, X_test):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # Identify numeric and categorical columns
    numeric_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns

    combined_df = pd.concat([X_train, X_test], ignore_index=True)

    # Standardize numeric columns
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        X_train[numeric_columns] = scaler.fit_transform(
            X_train[numeric_columns]
        )
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    # One-hot encode categorical columns
    if len(categorical_columns) > 0:
        # One-hot encode categorical columns based on the combined data
        encoder = OneHotEncoder(drop="first")
        X_combined_encoded = encoder.fit_transform(
            combined_df[categorical_columns]
        )

        # Split the combined encoded data back into training and testing sets
        X_train_encoded = X_combined_encoded[: len(X_train)]
        X_test_encoded = X_combined_encoded[len(X_train) :]

        # Concatenate encoded features with the original DataFrame
        X_train = pd.concat(
            [
                X_train.drop(categorical_columns, axis=1),
                pd.DataFrame(X_train_encoded.toarray()),
            ],
            axis=1,
        )
        X_test = pd.concat(
            [
                X_test.drop(categorical_columns, axis=1),
                pd.DataFrame(X_test_encoded.toarray()),
            ],
            axis=1,
        )

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    return X_train, X_test


def patch_nan(df):
    imputed_data = df.copy()

    # Identify categorical and numerical columns
    categorical_cols = imputed_data.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()
    numerical_cols = imputed_data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Fill numerical columns
    if numerical_cols:
        # Forward fill, then backward fill
        imputed_data[numerical_cols] = (
            imputed_data[numerical_cols].ffill().bfill()
        )
        # Fill any remaining NaNs with the mean
        imputed_data[numerical_cols] = imputed_data[numerical_cols].fillna(
            imputed_data[numerical_cols].mean()
        )

    # Fill categorical columns
    for col in categorical_cols:
        mode = imputed_data[col].mode()
        if not mode.empty:
            imputed_data[col] = imputed_data[col].fillna(mode.iloc[0])
        else:
            # If mode is not available (e.g., all values are NaN), fill with a default value
            imputed_data[col] = imputed_data[col].fillna("Unknown")

    # Ensure no NaN values are left
    assert (
        not imputed_data.isnull().values.any()
    ), "There are still NaN values in the DataFrame after imputation"

    return imputed_data


def delete_columns(df, columns):
    return df.drop(columns=columns, errors="ignore")


def map_object_types(df, columns, object_bool):
    if object_bool:
        return df.astype(dict(zip_longest(columns, [], fillvalue="object")))
    else:
        return df.astype(dict(zip_longest(columns, [], fillvalue="category")))


def load_aids(aids=None, object_bool=False, list_ids=[]):
    if aids is None:
        aids = pd.read_csv("RawData/aids_original_data.csv", sep=";").drop(
            columns="pidnum"
        )

    aids = aids.fillna({"cd496": aids.cd496.mean()})

    categorical_columns = [
        "hemo",
        "homo",
        "drugs",
        "karnof",
        "oprior",
        "z30",
        "zprior",
        "race",
        "gender",
        "str2",
        "strat",
        "symptom",
        "treat",
        "offtrt",
        "r",
        "cens",
        "arms",
    ]
    # aids["karnof"] = round(aids["karnof"]/10)*10
    # print("warning on import -> data loader")
    aids = delete_columns(aids, [])  # Adjust column names as needed
    aids = map_object_types(aids, categorical_columns, object_bool)
    aids = patch_nan(aids)

    if len(list_ids) > 0:
        aids = aids.iloc[list_ids]

    return aids


def load_wbcd(wbcd=None, object_bool=False, list_ids=[]):
    if wbcd is None:
        wbcd = pd.read_csv("RawData/breast_cancer_wisconsin.csv").drop(
            columns="Sample_code_number"
        )

    categorical_columns = list(wbcd.columns)

    wbcd = delete_columns(wbcd, [])  # Adjust column names as needed
    wbcd = map_object_types(wbcd, categorical_columns, object_bool)
    wbcd = patch_nan(wbcd)

    if len(list_ids) > 0:
        wbcd = wbcd.iloc[list_ids]

    return wbcd


def load_adult(adult=None, object_bool=False, list_ids=[]):
    if adult is None:
        adult = pd.read_csv("RawData/adult.csv")

    adult = delete_columns(
        adult, ["capital.loss", "fnlwgt"]
    )  # Adjust column names as needed

    categorical_columns = [
        "workclass",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
        "income",
    ]

    adult = adult.map(lambda x: np.nan if x == "?" else x)

    adult = map_object_types(adult, categorical_columns, object_bool)
    adult = patch_nan(adult)

    if len(list_ids) > 0:
        adult = adult.iloc[list_ids]

    return adult


def load_fewadult(fewadult=None, object_bool=False, list_ids=[]):
    if fewadult is None:
        from sklearn.model_selection import train_test_split

        adult = pd.read_csv("RawData/adult.csv")
        fewadult, _ = train_test_split(adult, test_size=0.5, random_state=42)
        fewadult.reset_index(inplace=True, drop=True)

    fewadult = delete_columns(
        fewadult, ["capital.loss", "fnlwgt"]
    )  # Adjust column names as needed

    categorical_columns = [
        "workclass",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
        "income",
    ]

    fewadult = fewadult.map(lambda x: np.nan if x == "?" else x)

    fewadult = map_object_types(fewadult, categorical_columns, object_bool)
    fewadult = patch_nan(fewadult)

    if len(list_ids) > 0:
        fewadult = fewadult.iloc[list_ids]

    return fewadult


def load_laws(laws=None, object_bool=False, list_ids=[]):
    if laws is None:
        laws = pd.read_csv("RawData/bar_pass_prediction.csv")

        columns_to_delete = [
            "ID",
            "other",
            "asian",
            "black",
            "hisp",
            "bar",
            "bar_passed",
            "index6040",
            "race2",
            "male",
            "sex",
            "grad",
            "Dropout",
            "race",
            "indxgrp",
            "indxgrp2",
            "gpa",
            "parttime",
            "decile1b",
            "cluster",
            "bar1",
            "bar1_yr",
            "bar2",
            "bar2_yr",
            "dnn_bar_pass_prediction",
        ]

        laws = delete_columns(laws, columns_to_delete)

    categorical_columns = ["fulltime", "fam_inc", "tier"]

    laws = map_object_types(laws, categorical_columns, object_bool)
    laws = patch_nan(laws)

    if len(list_ids) > 0:
        laws = laws.iloc[list_ids]

    return laws


def load_compas(compas=None, object_bool=False, list_ids=[]):
    if compas is None:
        compas = pd.read_csv("RawData/compas-scores.csv")

        columns_to_delete = [
            "id",
            "name",
            "first",
            "last",
            "age",
            "dob",
            "screening_date",
            "compas_screening_date",
            "c_jail_in",
            "c_jail_out",
            "c_case_number",
            "c_offense_date",
            "c_arrest_date",
            "days_b_screening_arrest",
            "r_offense_date",
            "num_r_cases",
            "r_case_number",
            "r_charge_degree",
            "r_days_from_arrest",
            "r_charge_desc",
            "r_jail_in",
            "r_jail_out",
            "is_violent_recid",
            "num_vr_cases",
            "vr_case_number",
            "vr_charge_degree",
            "vr_offense_date",
            "vr_charge_desc",
            "v_type_of_assessment",
            "v_decile_score",
            "v_score_text",
            "v_screening_date",
            "type_of_assessment",
            "decile_score.1",
            "decile_score",
            "score_text",
            "c_charge_desc",
        ]

        compas = delete_columns(compas, columns_to_delete)

    categorical_columns = [
        "sex",
        "age_cat",
        "race",
        "c_charge_degree",
        "is_recid",
    ]
    compas = compas.map(lambda x: np.nan if x == -1 else x)
    compas = map_object_types(compas, categorical_columns, object_bool)
    compas = patch_nan(compas)

    if len(list_ids) > 0:
        compas = compas.iloc[list_ids]

    return compas


def load_credit(credit=None, object_bool=False, list_ids=[]):
    if credit is None:
        credit = pd.read_csv("RawData/UCI_Credit_Card.csv")
        credit = delete_columns(
            credit, ["ID"]
        )  # Adjust column names as needed

    categorical_columns = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "default.payment.next.month",
    ]

    credit = map_object_types(credit, categorical_columns, object_bool)
    credit = patch_nan(credit)

    if len(list_ids) > 0:
        credit = credit.iloc[list_ids]

    return credit


def load_meps(meps=None, object_bool=False, list_ids=[]):
    if meps is None:
        meps = pd.read_csv("RawData/MEPS.csv")

    categorical_columns = [
        "REGION",
        "sex",
        "race",
        "MARRY",
        "FTSTU",
        "ACTDTY",
        "HONRDC",
        "RTHLTH",
        "MNHLTH",
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON",
        "CHOLDX",
        "CANCERDX",
        "DIABDX",
        "JTPAIN",
        "ARTHDX",
        "ARTHTYPE",
        "ASTHDX",
        "ADHDADDX",
        "PREGNT",
        "WLKLIM",
        "ACTLIM",
        "SOCLIM",
        "COGLIM",
        "DFHEAR42",
        "DFSEE42",
        "ADSMOK42",
        "K6SUM42",
        "PHQ242",
        "EMPST",
        "POVCAT",
        "INSCOV",
        "UTILIZATION",
    ]

    meps = map_object_types(meps, categorical_columns, object_bool)
    meps = patch_nan(meps)

    if len(list_ids) > 0:
        meps = meps.iloc[list_ids]

    return meps


def preload_path(pb):
    methods = [
        "CTGAN",
        "Synthpop",
        "Kanon",
        "Dataset",
    ]
    out = {
        method: loader_func(
            func_loader[pb], Path("Data", "{}_{}.csv".format(method, pb))
        )
        for method in methods
    }
    out["Dataset"] = loader_func(
        func_loader[pb], None
    )  # Example usage for the main dataset
    out["target"] = dict_targets[pb]
    return out


def loader_func(dataset_loader, path_file):
    def func(object_bool=False, list_ids=[]):
        if path_file is None:
            df = None
        else:
            df = pd.read_csv(path_file)
        return dataset_loader(df, object_bool, list_ids)

    return func


dict_targets = {
    "AIDS": "r",
    "WBCD": "Class",
    "LAWS": "pass_bar",
    "CREDIT": "default.payment.next.month",
    "MEPS": "UTILIZATION",
    "COMPAS": "is_recid",
    "ADULT": "income",
    "FEWADULT": "income",
}

func_loader = {
    "AIDS": load_aids,
    "WBCD": load_wbcd,
    "LAWS": load_laws,
    # "ADULT": load_adult,
    "FEWADULT": load_fewadult,
    "COMPAS": load_compas,
    "MEPS": load_meps,
    "CREDIT": load_credit,
}


dict_loader = {
    pb: preload_path(pb)
    for pb in ["AIDS", "WBCD", "LAWS", "FEWADULT", "CREDIT", "COMPAS", "MEPS"]
}


dict_xp_mia = {
    "AIDS": [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        23,
        24,
        25,
        27,
        28,
    ],
    "WBCD": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ],
    "LAWS": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    ],
    "COMPAS": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    ],
    "MEPS": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ],
    "ADULT": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    ],
    "FEWADULT": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    ],
    "CREDIT": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    ],
}


def generator(pb, gen_func, params=None):
    """
    Creates a generator function that encapsulates a given function (`gen_func`)
    and adds additional functionality by binding specific parameters.

    Args:
        pb: Str (name of the dataset) that the generator function (`gen_func`) uses.
        gen_func (callable): The generator function, takes `pb`, `seed_`, `list_ids`, and `params` as arguments.
        params (dict, optional): A dictionary of additional parameters to pass to `gen_func`.
            Defaults to an empty dictionary if not provided.

    Returns:
        callable: A function that accepts `seed_` and `list_ids` as arguments
        and invokes `gen_func` with `pb`, `seed_`, `list_ids`, and `params`.
    """
    # Ensure `params` is a dictionary, even if not explicitly provided
    if params is None:
        params = {}

    # Define the output function that wraps the `gen_func`
    def output_func(seed_=0, list_ids=None):
        """
        Wrapper function that invokes `gen_func` with predefined `pb` and `params`.

        Args:
            seed_ (int, optional): A seed value for the generator. Defaults to 0.
            list_ids (list, optional): A list of IDs to process. Defaults to an empty list if not provided.

        Returns:
            Any: The result of `gen_func` execution.
        """
        if list_ids is None:
            list_ids = []

        return gen_func(pb, seed_, list_ids, params)

    return output_func


def preload_subfuncs(loader, pb):
    dict_methods = {
        "CTGAN": generator(loader, CTGAN_generator),
        "Kanon": generator(
            loader, Kanon, {"k": 20, "target": dict_targets[pb]}
        ),
        "Dataset": generator(loader, Identity),
        "Avatar": None,
        "MST": generator(loader, MST, {"epsilon": 3}),
        "CompSAIPH": generator(loader, SAIPH_Compressor, {"nf": 5}),
        "CompSAIPH1": generator(loader, SAIPH_Compressor, {"nf": 1}),
        "CompSAIPH2": generator(loader, SAIPH_Compressor, {"nf": 2}),
        "CompSAIPH3": generator(loader, SAIPH_Compressor, {"nf": 3}),
        "CompSAIPH5": generator(loader, SAIPH_Compressor, {"nf": 5}),
        "CompSAIPH10": generator(loader, SAIPH_Compressor, {"nf": 10}),
        "CompSAIPH20": generator(loader, SAIPH_Compressor, {"nf": 20}),
        "MAvatar": generator(
            loader,
            MAvatar,
            {"size_ratio": 1, "min_bucket_size": 10, "max_dimensions": 5},
        ),
    }
    return dict_methods


def local_cloaking(X, Y):
    """
    Determines the index of the nearest point in `X` for each point in `Y`.

    Args:
        X (array-like of shape (n_samples_X, n_features)):
            Dataset containing the reference points.
        Y (array-like of shape (n_samples_Y, n_features)):
            Dataset containing the query points.

    Returns:
        numpy.ndarray of shape (n_samples_Y,):
            Array of indices indicating the nearest neighbor in `X` for each point in `Y`.
    """
    # Compute pairwise Euclidean distances between points in X and Y
    D = euclidean_distances(X, Y)

    # Find the index of the nearest neighbor in X for each point in Y
    return D.argsort(1).argmin(0)


def has_nan(data):
    """
    Check if there are NaN values in a dataset.

    Args:
    data (numpy.ndarray or pandas.DataFrame): The dataset to check for NaN values.

    Returns:
    bool: True if NaN values are found, False otherwise.
    """
    if isinstance(data, np.ndarray):
        return np.isnan(data).any()
    elif isinstance(data, pd.DataFrame):
        return data.isnull().values.any()
    else:
        raise ValueError(
            "Input data must be a numpy.ndarray or pandas.DataFrame"
        )


def gower_dist_matrix(A, B):
    """
    Computes the Gower distance matrix between two datasets, `A` and `B`.

    The Gower distance is a metric that supports mixed data types (numerical and categorical)
    and computes pairwise distances between observations in `A` and `B`.

    Args:
        A (pd.DataFrame):
            A DataFrame containing the first dataset.
        B (pd.DataFrame):
            A DataFrame containing the second dataset.

    Returns:
        numpy.ndarray:
            A distance matrix of shape (len(A), len(B)), where each element [i, j]
            represents the Gower distance between the i-th row of `A` and the j-th row of `B`.
    """
    # Combine datasets A and B into a single DataFrame
    combined_data = pd.concat([A, B], ignore_index=True)

    # Get the number of rows in A to separate distances later
    n = len(A)

    # Compute the Gower distance matrix for the combined dataset
    gower_dist = gower_matrix(combined_data)

    # Extract the distance sub-matrix for A vs B
    return gower_dist[:n, n:]


def selection_mia(dataset, seed=None):
    """
    Splits a dataset into two groups: `list_member` and `list_non_member`
    using a random shuffle. If a seed is provided, the shuffle is reproducible.

    Args:
        dataset (pd.DataFrame):
            The input dataset with an index used for splitting.
        seed (int, optional):
            A seed for the random number generator to ensure reproducibility.

    Returns:
        tuple:
            - list_member (list): A list of indices representing the "member" group.
            - list_non_member (list): A list of indices representing the "non-member" group.
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Convert dataset indices to a list and shuffle them randomly
    list_ids = list(dataset.index)
    random.shuffle(list_ids)

    # Split the indices into two roughly equal groups
    half_point = len(dataset) // 2
    list_member = list_ids[:half_point]  # First half as members
    list_non_member = list_ids[half_point:]  # Second half as non-members

    return list_member, list_non_member


def exp_plan_loader(pb, algo, seed):
    full_data = func_loader[pb]()
    if ("CompSAIPH" in algo) or ("MAvatar" in algo):
        with open(
            Path(
                "Data",
                "DataMIA",
                "{}splitMIA_{}_seed{}.pkl".format(algo, pb, seed),
            ),
            "rb",
        ) as pickle_file:
            split = pickle.load(pickle_file)
            list_member, list_non_member = split["train"], split["test"]
    else:
        seed_atk = dict_xp_mia[pb][seed]
        list_member, list_non_member = selection_mia(full_data, seed_atk)
    if algo == "Dataset":
        synth = func_loader[pb](full_data.loc[list_member])
    elif algo == "Control":
        synth = func_loader[pb](full_data.loc[list_non_member])
    # elif "MAvatar" == algo or "CompSAIPH" == algo:
    #     synth = func_loader[pb](
    #         pd.read_csv(
    #             Path("Data", "DataMIA", f"{algo}MIA_{pb}_seed{seed}.csv")
    #         )
    #     )
    elif algo == "Avatar":
        synth = func_loader[pb](
            pd.read_csv(
                Path(
                    "Data",
                    "DataOctopize",
                    f"AvatarMIA_{pb}_seed{seed_atk}.csv",
                )
            )
        )
    elif algo == "Synthpop":
        synth = func_loader[pb](
            pd.read_csv(Path("Data", "DataMIA", f"{algo}MIA_{pb}_{seed}.csv"))
        )
    else:
        synth = func_loader[pb](
            pd.read_csv(
                Path("Data", "DataMIA", f"{algo}MIA_{pb}_seed{seed}.csv")
            )
        )

    if algo == "Kanon":
        categorical_columns = synth.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        for col in categorical_columns:
            # Create the NumPy array
            a = synth[col].to_numpy()

            # Use np.char.split to split each element in the array
            split_array = np.char.split(a.astype(str), sep=",")

            # Convert the resulting array of lists into a 2D NumPy array
            synth[col] = np.array([item[0] for item in split_array])
            synth[col] = np.array(synth[col]).astype(type(full_data[col][0]))

        synth = func_loader[pb](synth)
    data = full_data.loc[list_member]
    control = full_data.loc[list_non_member]
    synth = patch_nan(synth)  # .reset_index(drop=True)
    data = patch_nan(data)  # .reset_index(drop=True)
    control = patch_nan(control)  # .reset_index(drop=True)

    return data, control, synth[data.columns]


dict_generator = {
    pb: preload_subfuncs(dict_loader[pb]["Dataset"], pb)
    for pb in func_loader.keys()
}
