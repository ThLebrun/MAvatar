import numpy as np
from pathlib import Path
import pickle
import pandas as pd
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import random

from sklearn.neighbors import NearestNeighbors


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from sklearn.preprocessing import OneHotEncoder

import saiph

from tqdm import tqdm


from scipy.spatial.distance import euclidean
import gower


from utils.utils import (
    exp_plan_loader,
    dict_loader,
    preprocess_double,
    func_loader,
)


list_metrics = [
    "Column Shapes",
    "Column Pair Trends",
    "Balanced Accuracy",
    "Hidden Rate",
    "Median Local Cloaking",
    "Singling Out",
    "Singling Out Multivariate",
    "Linkability",
    "Max Attribute Risk",
    "Mean Attribute Risk",
    "Median Attribute Risk",
]

dict_translator_sensitive = {
    "race": {
        "LAWS": "race1",
        "FEWADULT": "race",
        "ADULT": "race",
        "COMPAS": "race",
        "MEPS": "race",
        "AIDS": "race",
    },
    "gender": {
        "LAWS": "gender",
        "FEWADULT": "sex",
        "ADULT": "sex",
        "CREDIT": "SEX",
        "COMPAS": "sex",
        "MEPS": "sex",
        "AIDS": "gender",
    },
    "homosexuality": {"AIDS": "homo"},
    "drug use": {"AIDS": "drugs"},
}

list_race = list(dict_translator_sensitive["race"].keys())
list_gender = list(dict_translator_sensitive["gender"].keys())

translator = {
    "D_Bary": "Distance to Barycenter",
    "LOF": "Local Outlier Factor",
    "3NN": "Average Distance to 3 Nearest Neighbors",
    "1NN": "Distance to Nearest Neighbors",
    "Dist_displacement": "Displacement Norm",
    "0": "1st reduced axis",
    "1": "2nd reduced axis",
    "2": "3rd reduced axis",
    "3": "4th reduced axis",
    "4": "5th reduced axis",
    "Singling_Out_Multi": "Singling Out",
    "Linkability": "Linkablity",
    "Attribute_Risk": "Attribute Inference",
}


quantiles_pb = {
    "AIDS": [0, 50, 75, 90, 95, 99],
    "WBCD": [0, 50, 80, 90, 97.5],
    "COMPAS": [0, 50, 75, 90, 95, 99.75],
    "MEPS": [0, 50, 75, 90, 95, 99.9],
    "LAWS": [0, 50, 75, 90, 95, 99.9],
    "ADULT": [0, 50, 75, 90, 95, 99.9],
    "FEWADULT": [0, 50, 75, 90, 95, 99.9],
    "CREDIT": [0, 50, 75, 90, 95, 99.9],
}


isolation_approaches = [
    "Distance to Barycenter",
    "Isolation to Neighbors",
    "Both",
]


def rf_eval(df1, df2, target_column):
    from sklearn.ensemble import RandomForestClassifier

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    # # Combine both training and testing data to ensure consistent encoding
    # combined_df = pd.concat([df1, df2], ignore_index=True)
    # Separate features (X) and target variable (y) for training data
    X_train = df1.drop(target_column, axis=1)
    y_train = df1[target_column]

    # Separate features (X) and target variable (y) for testing data
    X_test = df2.drop(target_column, axis=1)
    y_test = df2[target_column]

    X_train, X_test = preprocess_double(X_train, X_test)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import balanced_accuracy_score

    # Calculate balanced accuracy
    return balanced_accuracy_score(y_test, y_pred)


def Singling_Out_Multi(pb, algo, seed):
    from anonymeter.evaluators import SinglingOutEvaluator

    ori, control, syn = exp_plan_loader(pb, algo, seed)
    dict_results = {}
    # if len(ori)>15000:
    #     print("reducinf datasets")
    #     ori = ori[:15000]
    #     syn = syn[:15000]
    n_attacks = 300
    n_cols = 4
    evaluator = SinglingOutEvaluator(
        ori=ori,
        syn=syn,
        control=control,
        n_attacks=n_attacks,  # this attack takes longer
        n_cols=n_cols,
    )

    try:
        evaluator.evaluate(mode="multivariate")
        risk = evaluator.risk()
        dict_results["Singling Out Multivariate"] = risk.value

    except RuntimeError as ex:
        print(
            f"Singling out evaluation failed with {ex}. Please re-run this cell."
            "For more stable results increase `n_attacks`. Note that this will "
            "make the evaluation slower."
        )
        dict_results["Singling Out Multivariate"] = np.nan
    print(evaluator.queries()[:3])
    return dict_results


def Linkability(pb, algo, seed):
    ori, control, syn = exp_plan_loader(pb, algo, seed)
    dict_results = {}

    def random_split(input_list, split_ratio=0.5, seed_=0):
        random.seed(seed_)
        random.shuffle(input_list)  # Shuffle the input_list in place
        split_index = int(len(input_list) * split_ratio)
        return input_list[:split_index], input_list[split_index:]

    from anonymeter.evaluators import LinkabilityEvaluator

    evaluator = LinkabilityEvaluator(
        ori=ori,
        syn=syn,
        control=control,
        n_attacks=min(2000, len(ori) // 5),
        aux_cols=random_split(list(ori.columns)),
        n_neighbors=10,
    )

    evaluator.evaluate(
        n_jobs=-2
    )  # n_jobs follow joblib convention. -1 = all cores, -2 = all execept one

    dict_results["Linkability"] = evaluator.risk(n_neighbors=10).value
    return dict_results


def Attribute_Risk(pb, algo, seed):
    data, control, synth = exp_plan_loader(pb, algo, seed)

    data.reset_index(inplace=True, drop=True)
    # Get the indices of the rows with the 10% highest values in the 'value' column
    results_global = attribute_risk_subfunc(data, control, synth, pb)
    return results_global


def attribute_risk_subfunc(ori_van, control_van, syn_van, pb):
    from anonymeter.evaluators import InferenceEvaluator

    dict_results = {}
    columns = ori_van.columns
    full_data = func_loader[pb]()
    ori = ori_van.copy()
    control = control_van.copy()
    syn = syn_van.copy()
    for col in columns:
        if pd.api.types.is_categorical_dtype(full_data[col]):
            ori[col] = ori[col].astype(
                pd.CategoricalDtype(categories=full_data[col].cat.categories)
            )
            control[col] = control[col].astype(
                pd.CategoricalDtype(categories=full_data[col].cat.categories)
            )
            syn[col] = syn[col].astype(
                pd.CategoricalDtype(categories=full_data[col].cat.categories)
            )
    if pb in "WBCD":
        results = []

        for secret in columns:
            aux_cols = [col for col in columns if col != secret]

            evaluator = InferenceEvaluator(
                ori=ori,
                syn=syn,
                control=control,
                aux_cols=aux_cols,
                secret=secret,
                n_attacks=min(1000, len(ori) // 5),
            )
            try:
                evaluator.evaluate(n_jobs=-2)
                results.append((secret, evaluator.results().risk().value))
            except TypeError as e:
                if (
                    str(e)
                    == "Categoricals can only be compared if 'categories' are the same."
                ):
                    print(
                        secret,
                        " : Categoricals can only be compared if 'categories' are the same",
                    )
                    # print(ori[secret], control[secret], syn[secret])
                    results.append((secret, np.nan))
                else:
                    raise e

        risks = np.array([res[1] for res in results])

        dict_results["Attributes Inference Risk"] = np.mean(risks)
        return dict_results
    elif pb == "CREDIT":
        columns_secret = ["gender"]
    elif pb in ["ADULT", "FEWADULT", "LAWS", "MEPS", "COMPAS"]:
        columns_secret = ["gender", "race"]
    elif pb == "AIDS":
        columns_secret = ["gender", "race", "homosexuality", "drug use"]
    else:
        return 1 / 0

    for secret in columns_secret:
        secret_column = dict_translator_sensitive[secret][pb]
        aux_cols = [col for col in columns if col != secret_column]
        print(secret_column)
        evaluator = InferenceEvaluator(
            ori=ori,
            syn=syn,
            control=control,
            aux_cols=aux_cols,
            secret=secret_column,
            n_attacks=min(1000, len(ori) // 5),
        )
        try:
            evaluator.evaluate(n_jobs=-2)
            risk_value = evaluator.results().risk().value
        except TypeError as e:
            if (
                str(e)
                == "Categoricals can only be compared if 'categories' are the same."
            ):
                print(
                    secret,
                    " : Categoricals can only be compared if 'categories' are the same",
                )
                print(
                    ori[secret_column],
                    control[secret_column],
                    syn[secret_column],
                )
                risk_value = np.nan
            else:
                raise e

        dict_results["{} Inference Risk".format(secret)] = copy(risk_value)
    return dict_results


def patch_gower(df):
    df_ = df.copy()
    cat_columns = df.select_dtypes(include=["category"]).columns
    df_[cat_columns] = df[cat_columns].astype("object")
    for col in df_.columns:
        if df_[col].dtype in ["int", "int64"]:
            df_[col] = df_[col].astype("float64")
    return df_


def AIA_Attack(df_real, df_synth, col_sensitive, focus_ids=None):
    ##Approach SAIPH
    #     import saiph
    #     if focus_ids is None:
    #         focus_ids = np.arange(len(df_real))
    #     encoder = OneHotEncoder(sparse_output=False)  # drop='first' to avoid dummy variable trap
    #     df_attacked = df_real.drop(columns = [col_sensitive])
    #     df_synth_attacked = df_synth.drop(columns = [col_sensitive])
    #     sensitive_occurences = np.unique(df_real[[col_sensitive]].to_numpy())
    #     if len(sensitive_occurences)!=2:
    #         1/0

    #     encoded_data_sensitive = encoder.fit_transform(df_real[[col_sensitive]])
    #     encoded_synt_sensitive = encoder.transform(df_synth[[col_sensitive]])

    #     coord_data, model = saiph.fit_transform(df_attacked)
    #     coord_data = coord_data.to_numpy()[focus_ids,:]
    #     coord_synth = saiph.transform(df_synth_attacked, model).to_numpy()[:,:]

    #     from sklearn.neighbors import NearestNeighbors

    #     # Fit NearestNeighbors model on B
    #     nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(coord_synth)

    #     # Find the indices of the closest points in B for each point in A
    #     _, indices = nearest_neighbors.kneighbors(coord_data)

    # ##Approach Gower
    # import numpy as np
    # from sklearn.neighbors import NearestNeighbors
    # import gower

    # if focus_ids is None:
    #     focus_ids = np.arange(len(df_real))

    # # One-hot encode the sensitive column

    # df_attacked = df_real.drop(columns=[col_sensitive])
    # df_synth_attacked = df_synth.drop(columns=[col_sensitive])
    # sensitive_occurences = np.unique(df_real[[col_sensitive]].to_numpy())

    # if len(sensitive_occurences) != 2:
    #     raise ValueError("Sensitive column must have exactly 2 unique values.")

    # encoder = OneHotEncoder(sparse_output=False)
    # encoded_data_sensitive = encoder.fit_transform(df_real[[col_sensitive]])
    # encoded_synth_sensitive = encoder.transform(df_synth[[col_sensitive]])

    # # Compute the Gower distance matrix between df_real and df_synth
    # distances = gower.gower_matrix(patch_gower(df_attacked.iloc[focus_ids, :]),
    #                                      patch_gower(df_synth_attacked))

    # # Fit NearestNeighbors model using the Gower distance
    # # nearest_neighbors = NearestNeighbors(n_neighbors=1, metric='precomputed').fit(gower_distances)

    # # # Find the indices of the closest points in df_synth for each point in df_real (restricted by focus_ids)
    # # distances, indices = nearest_neighbors.kneighbors(gower_distances)

    ##Aproach FAMD
    from sklearn.decomposition import PCA

    if focus_ids is None:
        focus_ids = np.arange(len(df_real))
    encoder = OneHotEncoder(
        sparse_output=False
    )  # drop='first' to avoid dummy variable trap
    df_attacked = df_real.drop(columns=[col_sensitive])
    df_synth_attacked = df_synth.drop(columns=[col_sensitive])
    sensitive_occurences = np.unique(df_real[[col_sensitive]].to_numpy())
    if len(sensitive_occurences) != 2:
        1 / 0

    encoded_data_sensitive = encoder.fit_transform(df_real[[col_sensitive]])
    encoded_synth_sensitive = encoder.transform(df_synth[[col_sensitive]])

    # Fit FAMD on the dataset
    pca = PCA()
    pca.fit(df_attacked)
    # Transform the dataset to its principal components
    coord_data = (
        pd.DataFrame(pca.transform(df_attacked), index=df_real.index)
        .iloc[focus_ids, :]
        .to_numpy()
    )
    coord_synth = pca.transform(df_synth_attacked)

    from sklearn.neighbors import NearestNeighbors

    # Fit NearestNeighbors model on B
    nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        coord_synth
    )

    # Find the indices of the closest points in B for each point in A
    _, indices = nearest_neighbors.kneighbors(coord_data)
    # indices = np.argmin(gower_distances, axis = 1)
    sensitive_guess = encoded_synth_sensitive[indices][:, 0, 0]  # [:,0,0]
    sensitive_truth = encoded_data_sensitive[focus_ids, 0]

    return balanced_accuracy_score(sensitive_truth, sensitive_guess)


def perf_SDV(data1, data2):
    from sdv.metadata import SingleTableMetadata
    from sdmetrics.reports.single_table import QualityReport

    metadata_object = SingleTableMetadata()

    metadata_object.detect_from_dataframe(data=data1)
    metadata = metadata_object.to_dict()
    report = QualityReport()
    report.generate(data1, data2, metadata)

    perfs = report.get_properties()
    return {perfs.Property[i]: perfs.Score[i] for i in range(len(perfs))}


def SDV(pb, algo, seed):
    dict_results = {}
    data, _, syn = exp_plan_loader(pb, algo, seed)
    data.reset_index(inplace=True, drop=True)
    syn.reset_index(inplace=True, drop=True)
    perf = perf_SDV(data, syn)
    dict_results["Column Shapes"] = perf["Column Shapes"]
    dict_results["Column Pair Trends"] = perf["Column Pair Trends"]
    return dict_results


def Bal_Accuracy(pb, algo, seed):
    dict_results = {}
    _, control, syn = exp_plan_loader(pb, algo, seed)
    control = control.reset_index(drop=True)
    syn = syn.reset_index(drop=True)
    target = dict_loader[pb]["target"]
    dict_results["Balanced Accuracy"] = rf_eval(syn, control, target)
    return dict_results


def func_mia_seed(params):
    (
        pp_inversed_data,
        pp_data_atk,
        Isolation,
        data_member,
    ) = prepare_attack_plan(params)
    list_ids = list(pp_data_atk.index)
    list_member = list(data_member.index)
    result_df = pd.DataFrame(Isolation, columns=["Isolation"], index=list_ids)
    result_df["MIA"] = np.array(
        [int(id_point in list_member) for id_point in list_ids]
    ).astype(bool)
    _, ind_neigh_atk = find_nearest_neighbors(
        pp_inversed_data, pp_data_atk, True
    )  # dist_neigh_atk

    for n_hit in np.arange(1, 21):
        hitten_neighbors = ind_neigh_atk.to_numpy()[:, :n_hit].flatten()
        count_hit = {id_neighbor: 0 for id_neighbor in list_ids}
        # selected_neighbors = np.argwhere(dist_neightbors<params["size_sphere"])[:,0]
        # hitten_neighbors = hitten_neighbors[selected_neighbors]

        for id_neigh in hitten_neighbors:
            count_hit[id_neigh] += 1
        result_df[f"n_hit{n_hit}"] = np.array(list(count_hit.values()))
    return result_df


def MIA_Attack(params):
    pb = params["pb"]
    result = {}
    print(pb, params["algo"])
    if params["algo"] == "CompSAIPH":
        print(pb, params["algo"])
        for seed in range(params["range_seeds"]):
            try:
                params["seed"] = seed
                res = func_mia_seed(params)
            except:
                res = np.NAN
            finally:
                result[seed] = deepcopy(res)
    else:
        for seed in range(params["range_seeds"]):
            params["seed"] = seed
            result[seed] = func_mia_seed(params)
    return result


def MIA_Analysis(results, params):
    pb = params["pb"]
    precision_mat = np.ones((len(quantiles_pb[pb]), 25, 20)) * -1
    recall_mat = np.ones((len(quantiles_pb[pb]), 25, 20)) * -1
    accuracy_mat = np.ones((len(quantiles_pb[pb]), 25, 20)) * -1
    f1_mat = np.ones((len(quantiles_pb[pb]), 25, 20)) * -1
    ba_mat = np.ones((len(quantiles_pb[pb]), 25, 20)) * -1
    auc_mat = np.ones((len(quantiles_pb[pb]), 25, 20)) * -1
    print(pb, params["algo"])
    for seed in range(params["range_seeds"]):
        params["seed"] = seed
        result = results[seed]
        list_datasets = split_dataframe_by_quantiles(
            result, quantile_list=quantiles_pb[params["pb"]]
        )

        for i_data, sub_data in enumerate(list_datasets):
            for i_hit, n_hit in enumerate(range(1, 21)):
                ids = list(sub_data.index)
                count_hit = sub_data[f"n_hit{n_hit}"]
                median_hit = np.median(
                    [count_hit.loc[id_number] for id_number in ids]
                )
                # guess = [count_hit[id_number]>=median_hit for id_number in ids]
                guess = [
                    count_hit.loc[id_number] > median_hit for id_number in ids
                ]

                values_hit = {
                    id_point: count_hit.loc[id_point] for id_point in ids
                }

                truth = sub_data["MIA"]

                precision_mat[i_data, seed, i_hit] = precision_score(
                    truth, guess, zero_division=0.0
                )
                recall_mat[i_data, seed, i_hit] = recall_score(truth, guess)
                accuracy_mat[i_data, seed, i_hit] = accuracy_score(
                    truth, guess
                )
                f1_mat[i_data, seed, i_hit] = f1_score(truth, guess)
                ba_mat[i_data, seed, i_hit] = balanced_accuracy_score(
                    truth, guess
                )
                auc_mat[i_data, seed, i_hit] = roc_auc_score(
                    truth, np.array(list(values_hit.values()))
                )
    return {
        "precision": precision_mat,
        "recall": recall_mat,
        "accuracy": accuracy_mat,
        "f1": f1_mat,
        "ba": ba_mat,
        "auc": auc_mat,
    }


def MIA_builder(pb, algo):
    params = {
        "algo": algo,
        "pb": pb,
        "nf": 5,
        "range_seeds": 25,
    }
    results = MIA_Attack(params)
    pickle.dump(
        results,
        open(
            Path("Metrics", "MIA", f"Data_{pb}{algo}.pickle"),
            "wb",
        ),
    )
    results = pickle.load(
        open(
            Path("Metrics", "MIA", f"Data_{pb}{algo}.pickle"),
            "rb",
        )
    )
    result = MIA_Analysis(results, params)
    pickle.dump(
        result,
        open(
            Path("Metrics", "MIA", f"Result_{pb}{algo}.pickle"),
            "wb",
        ),
    )
    result = pickle.load(
        open(
            Path("Metrics", "MIA", f"Result_{pb}{algo}.pickle"),
            "rb",
        )
    )


def MIA_exploiter(pb, algo, seed):
    result = pickle.load(
        open(
            Path("Metrics", "MIA", f"Result_{pb}{algo}.pickle"),
            "rb",
        )
    )
    data = result["ba"]
    results_out = np.zeros(data.shape[0])
    # for seed in range(25):
    train_seeds = list(range(25))
    train_seeds.remove(seed)
    n_filters = np.argmax(np.mean(data[:, train_seeds, :], axis=1), axis=1)
    for i_quantile, n_filter in enumerate(n_filters):
        results_out[i_quantile] = data[i_quantile, seed, n_filter]
    # perf_by_quantiles = np.mean(results_out, axis = 1)
    # avg_perf = np.mean(perf_by_quantiles)
    avg_perf = np.mean(results_out)
    return {"MIA": avg_perf, "MIA_Outlier": results_out[-1]}


dict_metrics = {
    "Singling_Out_Multi": Singling_Out_Multi,
    "Linkability": Linkability,
    "Attribute_Risk": Attribute_Risk,
    "SDV": SDV,
    "Bal_Accuracy": Bal_Accuracy,
    "MIA": MIA_exploiter,
}


def sensitive_cols(pb_):
    if pb_ in ["WBCD"]:
        return ["Attributes Inference Risk"]
    elif pb_ == "CREDIT":
        return ["gender Inference Risk"]
    elif pb_ in ["ADULT", "LAWS", "MEPS", "COMPAS", "FEWADULT"]:
        return ["gender Inference Risk", "race Inference Risk"]
    elif pb_ in ["AIDS"]:
        return [
            "gender Inference Risk",
            "race Inference Risk",
            "homosexuality Inference Risk",
            "drug use Inference Risk",
        ]
    return 1 / 0


def sensitive_cols_NN(pb_):
    list_attributes = sensitive_cols(pb_)
    list_out = list()
    for attribute in list_attributes:
        list_out.append(f"{attribute} NN")
    return list_out


trad_metrics = {
    "SDV": lambda x: ["Column Shapes", "Column Pair Trends"],
    "Bal_Accuracy": lambda x: ["Balanced Accuracy"],
    # "OctoPrivacy":lambda x:["Hidden Rate", "Median Local Cloaking"],
    "Singling_Out_Multi": lambda x: ["Singling Out Multivariate"],
    "Linkability": lambda x: ["Linkability"],
    "Attribute_Risk": sensitive_cols,
    "MIA": lambda x: ["MIA", "MIA_Outlier"],
}


def open_metrics_dict(path_name, algo=None):
    try:
        output = pickle.load(open(path_name, "rb"))
    except:
        print("No elemenent found on this path, creating new dictionary")
        output = dict()
    if algo not in list(output.keys()) and (algo is not None):
        output[algo] = dict()
    return output


def compute_metric(data_name, metric_name, algo_name):
    modif = {}
    if metric_name == "MIA":
        MIA_builder(data_name, algo_name)

    for key_ in trad_metrics[metric_name](data_name):
        modif[key_] = list(np.ones(25) * -1)
    for seed in range(25):
        res_out = dict_metrics[metric_name](data_name, algo_name, seed)
        for key_ in trad_metrics[metric_name](data_name):
            modif[key_][seed] = copy(res_out[key_])

    print("{} {} {} computed".format(data_name, algo_name, metric_name))

    for metric in trad_metrics[metric_name](data_name):
        pickle.dump(
            modif[metric],
            open(
                Path("Metrics", f"{data_name}_{algo_name}_{metric}.pkl"), "wb"
            ),
        )
    return modif


def save_values_to_txt(list_methods, results_dict, pb_):
    """
    Usefull to quickly create LateX Tabulars
    """
    pack_metrics = [trad_metrics[key_](pb_) for key_ in trad_metrics.keys()]

    methods = [s.replace("_", " ") for s in list_methods]
    methods[0] = "R. Avatar 1"
    methods[1] = "R. Avatar 2"
    methods[-1] = "Identity"

    with open(Path("SaveMetrics", f"Metrics_{pb_}.tex"), "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\small\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{ccccccc}\n")
        f.write("\\toprule\n")
        f.write("\\midrule\n")
        f.write(
            f"{pb_} & {methods[0]} & {methods[1]} & {methods[2]} & {methods[3]} & {methods[4]} & {methods[5]} \\\\ \\midrule\n"
        )
        f.write("\\bottomrule\n")
        for metrics in pack_metrics:
            for metric in metrics:
                means = [
                    np.mean(results_dict[method_name][metric])
                    for method_name in list_methods
                ]
                f.write(
                    f"{metric} & {means[0]:.3f} & {means[1]:.3f} & {means[2]:.3f} & {means[3]:.3f} & {means[4]:.3f} & {means[5]:.3f} \\\\ \\midrule\n"
                )
            f.write("\\bottomrule\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def compute_center(df, distance_metric="euclidean"):
    if distance_metric not in ["euclidean", "gower"]:
        raise ValueError(
            "Unsupported distance metric. Use 'euclidean' or 'gower'."
        )

    if distance_metric == "euclidean":
        # For Euclidean distance, compute the mean for each column
        center = df.mean(axis=0)
    else:
        # For Gower distance, compute the mean for numerical columns
        # and the mode (most frequent value) for categorical columns
        center = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                center.append(df[col].mean())
            else:
                center.append(df[col].mode()[0])
        center = pd.Series(center, index=df.columns)

    return center


def distance_to_center(df, distance_metric="euclidean"):
    center = compute_center(df, distance_metric)

    if distance_metric == "euclidean":
        # Calculate Euclidean distances from each point to the center
        distances = df.apply(lambda row: euclidean(row, center), axis=1)
        distances = distances.to_numpy()
    else:
        # Calculate Gower distances from each point to the center
        df_ = df.copy()
        cat_columns = df.select_dtypes(include=["category"]).columns
        df_[cat_columns] = df[cat_columns].astype("object")
        for col in df_.columns:
            if df_[col].dtype in ["int", "int64"]:
                df_[col] = df_[col].astype("float64")
        center_df = pd.DataFrame([center], columns=df_.columns)
        distances = gower.gower_matrix(df_, center_df)

    return distances


def find_nearest_neighbors(df1, df2, distances_bool=False):
    nn_model = NearestNeighbors(n_neighbors=len(df1)).fit(df2)

    df1_ = pd.DataFrame(df1.to_numpy(), columns=df2.columns, index=df1.index)
    # Find the indices of nearest neighbors for each row in df1
    distances, nearest_neighbors_indices = nn_model.kneighbors(
        df1_, return_distance=True
    )

    nearest_neighbors_index = df2.index.to_numpy()[nearest_neighbors_indices]

    if distances_bool:
        return distances, pd.DataFrame(
            nearest_neighbors_index, index=df1.index
        )
    else:
        return pd.DataFrame(nearest_neighbors_index, index=df1.index)


def sample_random_seeds(random_seeds, value_to_remove, size=5):
    # Remove the specific value
    random_seeds_ = deepcopy(random_seeds)
    random_seeds_.remove(value_to_remove)

    random.seed(42)

    sampled_seeds = random.sample(random_seeds_, size)

    return sampled_seeds


def get_curve(df, col, x):
    return stats.gaussian_kde(np.array(df[col]))(x)


def name_quantiles(quantile_list):
    out = []
    quantile_list_ = quantile_list + [100]
    for i_quantile, quant_name in enumerate(quantile_list_[:-1]):
        out.append(f"{quant_name}-{quantile_list_[i_quantile+1]}%")
    return out


def split_dataframe_by_quantiles(
    df, column_name="Isolation", quantile_list=[0, 10, 30, 70, 80, 90, 95]
):
    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=column_name)
    df_list = []

    # Iterate over the quantile ranges
    for i in range(len(quantile_list) - 1):
        lower_quantile = quantile_list[i]
        upper_quantile = quantile_list[i + 1]

        # # Calculate the quantile values
        # lower_value = df_sorted[column_name].quantile(lower_quantile / 100)
        # upper_value = df_sorted[column_name].quantile(upper_quantile / 100)

        # # Filter the DataFrame based on the quantile range
        # filtered_df = df_sorted[(df_sorted[column_name] >= lower_value) & (df_sorted[column_name] < upper_value)]
        low_id = round((lower_quantile / 100) * len(df_sorted))
        up_id = round((upper_quantile / 100) * len(df_sorted))
        df_list.append(df_sorted.iloc[low_id:up_id, :])
    df_list.append(df_sorted.iloc[up_id:, :])
    return df_list


def topo_euclid(df):
    df_out = df.copy()
    mean = pd.DataFrame([df.mean()])
    df_ = pd.concat([df, mean], ignore_index=True)

    # cat_columns = df.select_dtypes(include=['category']).columns
    # df_[cat_columns] = df[cat_columns].astype('object')
    for i_col, col in enumerate(df_.columns):
        if df_[col].dtype in ["int", "int64", "category", "object"]:
            df_[col] = df_[col].astype("float64")
    full_dist = squareform(pdist(df_))

    df_out["D_Bary"] = full_dist[-1, :-1]
    dist = full_dist[:-1, :-1]  # removing distance with center
    dist = np.partition(dist, 4)[:, :4]
    dist.sort()
    dist = dist[:, 1:4]
    df_out["3NN"] = np.mean(dist, axis=1)
    df_out["1NN"] = np.min(dist, axis=1)
    return df_out.astype(float)


def compute_datacenter(df):
    center = {}
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            center[col] = df[col].mode()[0]  # Mode for categorical columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            center[col] = df[col].mean()  # Mean for numeric columns
        else:
            center[col] = df[col].iloc[
                0
            ]  # Take the first value for other types
    return pd.DataFrame(center, index=[0])


def gower_distance_center(df):
    from gower import gower_matrix

    center = compute_datacenter(df)
    # Calculate Gower distance matrix
    df_ = pd.concat([center, df])

    cat_columns = df.select_dtypes(include=["category"]).columns
    df_[cat_columns] = df[cat_columns].astype("object")
    for col in df_.columns:
        if df_[col].dtype in ["int", "int64"]:
            df_[col] = df_[col].astype("float64")
    gower_dist_matrix = gower_matrix(df_)

    return gower_dist_matrix


def topo_gower(df):
    df_out = df.copy()

    full_dist = gower_distance_center(df)

    df_out["D_Bary"] = full_dist[0, 1:]
    dist = full_dist[:-1, :-1]  # removing distance with center
    dist = np.partition(dist, 4)[:, :4]
    dist.sort()
    dist = dist[:, 1:4]
    df_out["3NN"] = np.mean(dist, axis=1)
    df_out["1NN"] = np.min(dist, axis=1)
    return df_out


def remove_character(input_string, char_to_remove):
    return input_string.replace(char_to_remove, "")


def replace_substrings(input_string, replacements):
    for old_substring, new_substring in replacements:
        input_string = input_string.replace(old_substring, new_substring)
    return input_string


def visualize_redspace(visu, arrays_visu, name, save_name):
    if visu:
        # Create a seaborn figure
        plt.figure(figsize=(8, 6))

        for visu_array in arrays_visu:
            plt.scatter(
                np.array(visu_array[0])[:, 0],
                np.array(visu_array[0])[:, 1],
                color=visu_array[2],
                marker=visu_array[3],
                label=visu_array[1],
                s=visu_array[4],
            )

        # Add labels and title
        plt.xlabel("First axis")
        plt.ylabel("Second axis")
        plt.title(name)

        # Add legend
        plt.legend()
        # plt.savefig(Path("Plots", "DistribDist", f"{save_name}_Original_{pb}.png"))

        # Show plot
        plt.show()
    else:
        pass


def visualize_redspace_large(
    visu, arrays_visu, name, save_name, visu_arrows=None
):
    if visu:
        # Create a seaborn figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for visu_array in arrays_visu:
            axes[0, 0].scatter(
                np.array(visu_array[0])[:, 0],
                np.array(visu_array[0])[:, 1],
                color=visu_array[2],
                marker=visu_array[3],
                label=visu_array[1],
                s=visu_array[4],
            )
            axes[0, 1].scatter(
                np.array(visu_array[0])[:, 1],
                np.array(visu_array[0])[:, 2],
                color=visu_array[2],
                marker=visu_array[3],
                label=visu_array[1],
                s=visu_array[4],
            )
            axes[1, 0].scatter(
                np.array(visu_array[0])[:, 2],
                np.array(visu_array[0])[:, 3],
                color=visu_array[2],
                marker=visu_array[3],
                label=visu_array[1],
                s=visu_array[4],
            )
            axes[1, 1].scatter(
                np.array(visu_array[0])[:, 3],
                np.array(visu_array[0])[:, 4],
                color=visu_array[2],
                marker=visu_array[3],
                label=visu_array[1],
                s=visu_array[4],
            )

        axes[0, 0].set_xlabel("First axis")
        axes[0, 1].set_xlabel("Second axis")
        axes[1, 0].set_xlabel("Third axis")
        axes[1, 1].set_xlabel("Fourth axis")
        axes[0, 0].set_ylabel("Second axis")
        axes[0, 1].set_ylabel("Third axis")
        axes[1, 0].set_ylabel("Fourth axis")
        axes[1, 1].set_ylabel("Fifth axis")

        if visu_arrows is not None:
            for i in range(len(visu_arrows)):
                axes[0, 0].arrow(
                    visu_arrows[i, 0, 0],
                    visu_arrows[i, 0, 1],
                    0.85 * (visu_arrows[i, 1, 0] - visu_arrows[i, 0, 0]),
                    0.85 * (visu_arrows[i, 1, 1] - visu_arrows[i, 0, 1]),
                    color="black",
                    head_width=0.2,
                    head_length=0.2,
                )
                axes[0, 1].arrow(
                    visu_arrows[i, 0, 1],
                    visu_arrows[i, 0, 2],
                    0.85 * (visu_arrows[i, 1, 1] - visu_arrows[i, 0, 1]),
                    0.85 * (visu_arrows[i, 1, 2] - visu_arrows[i, 0, 2]),
                    color="black",
                    head_width=0.2,
                    head_length=0.2,
                )
                axes[1, 0].arrow(
                    visu_arrows[i, 0, 2],
                    visu_arrows[i, 0, 3],
                    0.85 * (visu_arrows[i, 1, 2] - visu_arrows[i, 0, 2]),
                    0.85 * (visu_arrows[i, 1, 3] - visu_arrows[i, 0, 3]),
                    color="black",
                    head_width=0.2,
                    head_length=0.2,
                )
                axes[1, 1].arrow(
                    visu_arrows[i, 0, 3],
                    visu_arrows[i, 0, 4],
                    0.85 * (visu_arrows[i, 1, 3] - visu_arrows[i, 0, 3]),
                    0.85 * (visu_arrows[i, 1, 4] - visu_arrows[i, 0, 4]),
                    color="black",
                    head_width=0.2,
                    head_length=0.2,
                )
        plt.title(name)

        # Add legend
        plt.legend()
        # plt.savefig(Path("Plots", "DistribDist", f"{save_name}_Original_{pb}.png"))

        # Show plot
        plt.show()
    else:
        pass


def prepare_attack_plan(params):
    from utils.Generators.SAIPH_Compressor import ToCategoricalProcessor

    data_member, data_non_member, synth_atk = exp_plan_loader(
        params["pb"], params["algo"], params["seed"]
    )
    data_aux = func_loader[params["pb"]]()

    processor = ToCategoricalProcessor(to_categorical_threshold=20)
    processed_records_aux = processor.preprocess(data_aux)

    projections_records_aux_recomputed, model = saiph.fit_transform(
        processed_records_aux
    )
    projections_records_aux_recomputed = (
        projections_records_aux_recomputed.iloc[:, : params["nf"]]
    )

    pp_data_aux = projections_records_aux_recomputed.copy()
    pp_synth_atk = compute_saiph(synth_atk, processor, model, params["nf"])
    pp_data_atk = pp_data_aux  # (
    #    pp_data_aux.copy()
    # )  # compute_saiph(data_aux.loc[list_member], processor, model, params["nf"])

    d_center = distance_to_center(pp_data_aux, "euclidean")

    return pp_synth_atk, pp_data_atk, d_center, data_member  # list_dataset


def func_reid_seed(params):
    (
        pp_inversed_data,
        pp_data_atk,
        d_center_redu,
        data_member,
    ) = prepare_attack_plan(params)

    list_member = list(data_member.index)
    pp_data_atk = pp_data_atk.iloc[data_member.index]
    size = len(pp_inversed_data)
    pp_inversed_data.set_index([list_member], drop=True, inplace=True)
    ind_neigh_atk = find_nearest_neighbors(
        pp_data_atk, pp_inversed_data
    )  # verify here
    ind_neigh_atk = ind_neigh_atk.to_numpy()
    indexes = np.repeat(np.array(list_member), size).reshape(
        len(list_member), size
    )  # Example list of indexes

    a = ind_neigh_atk - indexes
    where_ids = np.ones(indexes.shape[0]) * len(pp_data_atk)
    found = np.argwhere(a == 0)
    where_ids[found[:, 0]] = found[:, 1]

    output_LC = pd.DataFrame(
        -1 * np.ones((len(pp_inversed_data), 3)),
        index=data_member.index,
        columns=["DBaryFull", "DBaryRed", "LC"],
    )
    output_LC["DBaryRed"] = distance_to_center(pp_data_atk, "euclidean")
    output_LC["DBaryFull"] = distance_to_center(data_member, "gower")
    output_LC["LC"] = where_ids

    return output_LC


def Reid_Attack(params):
    local_cloaking = {
        f"{seed}": np.NAN for seed in range(params["range_seeds"])
    }
    for seed in tqdm(range(params["range_seeds"])):
        params["seed"] = seed
        local_cloaking[f"{seed}"] = func_reid_seed(params)
    return local_cloaking


def plot_curves(curves, parameters, target, title_):
    studied_quantiles = quantiles_pb[parameters["pb"]]

    for curv in curves:
        plt.plot(
            np.mean(curv[0][target], axis=1),
            label=curv[1],
            ls=curv[2],
            color=curv[3],
        )

    random_guess = np.ones(len(studied_quantiles)) * 0.5

    plt.plot(random_guess, label="Random Guess", color="grey", ls=":")

    names_qtiles = name_quantiles(studied_quantiles)
    plt.xticks(range(len(studied_quantiles)), names_qtiles, rotation=45)

    # bottom_line = min(np.concatenate((accuracy_list, ba_list, f1_list, precision_list, recall_list, random_guess)))

    # # Add text note to the bottom right corner
    # plt.text(len(studied_quantiles)*1.1, bottom_line, f'1% is {size//100} points\nlast quantile is {round(size*(1-quantiles_pb[parameters["pb"]][-1]/100))} points', ha='right', va='bottom')
    # # Adding labels, legend, and grid
    plt.xlabel(f'{parameters["pb"]} Quantiles of Distance to Barycenter')
    plt.ylabel("Score")
    # plt.ylim(0,1)
    plt.title(
        f'{title_} Over Quantiles for {curv[0][target].shape[1]} MIA Attacks - {parameters["pb"]}'
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.grid(True)
    # plt.savefig(Path("Plots", "MIA_Inverser", f"{pb}.png"))
    plt.show()
    print(parameters["pb"])
    print("--------")


def plot_curves_MIA(curves, parameters, target, title_, save=False):
    studied_quantiles = quantiles_pb[parameters["pb"]]

    for curv in curves:
        plt.plot(
            np.mean(curv[0][target], axis=1),
            label=curv[1],
            ls=curv[2],
            color=curv[3],
        )

    random_guess = np.ones(len(studied_quantiles)) * 0.5

    plt.plot(
        random_guess, label="Random Guess", color="grey", ls="-.", linewidth=2
    )

    names_qtiles = name_quantiles(studied_quantiles)
    plt.xticks(range(len(studied_quantiles)), names_qtiles, rotation=45)

    # bottom_line = min(np.concatenate((accuracy_list, ba_list, f1_list, precision_list, recall_list, random_guess)))

    # # Add text note to the bottom right corner
    # plt.text(len(studied_quantiles)*1.1, bottom_line, f'1% is {size//100} points\nlast quantile is {round(size*(1-quantiles_pb[parameters["pb"]][-1]/100))} points', ha='right', va='bottom')
    # # Adding labels, legend, and grid
    plt.xlabel(f'{parameters["pb"]} Quantiles of Distance to Barycenter')
    plt.ylabel(f"{target} Score")
    # plt.ylim(0,1)
    plt.legend(loc="lower left")
    plt.grid(True)
    if save:
        plt.savefig(
            Path(
                "Plots",
                f"{parameters['pb']}_MIA_{target}.png",
            )
        )
    plt.show()
    print(parameters["pb"])
    print("--------")


def plot_reidentification(results, parameters, FullSpace=False, save=False):
    if FullSpace:
        iso, name = "DBaryFull", "full"
    else:
        iso, name = "DBaryRed", "red"
    n_quantile = len(quantiles_pb[parameters["pb"]])
    split_data = {f"{i_quantile}": [] for i_quantile in range(n_quantile)}
    for seed in range(25):
        list_datasets = split_dataframe_by_quantiles(
            results[str(seed)],
            column_name=iso,
            quantile_list=quantiles_pb[parameters["pb"]],
        )
        for i_data, sub_data in enumerate(list_datasets):
            split_data[str(i_data)].append(sub_data["LC"].to_numpy())
    reid = np.array(
        [
            [
                np.mean(np.array(list(split_data[str(i_quantile)][seed])) == 0)
                for seed in range(25)
            ]
            for i_quantile in range(n_quantile)
        ]
    )

    means = np.mean(reid, axis=1)
    quantiles_02 = np.quantile(reid, 0.2, axis=1)
    quantiles_08 = np.quantile(reid, 0.8, axis=1)
    plt.figure(figsize=(6, 5), dpi=80)
    plt.plot(range(reid.shape[0]), means, label="Mean", color="CornFlowerBlue")
    plt.plot(
        range(reid.shape[0]), quantiles_02, color="CornFlowerBlue", ls="--"
    )
    plt.plot(
        range(reid.shape[0]), quantiles_08, color="CornFlowerBlue", ls="--"
    )
    plt.fill_between(
        range(reid.shape[0]),
        quantiles_02,
        quantiles_08,
        color="CornFlowerBlue",
        alpha=0.5,
        label="Quantile 0.2-0.8",
    )

    # plt.title(f'Reidentification Metrics Over Quantiles for {len(res["0"].keys())} Attacks - {parameters["pb"]}')

    plt.ylabel("Reidentification", fontsize=12)

    studied_quantiles = quantiles_pb[parameters["pb"]]
    size = len(func_loader[parameters["pb"]]())
    random_guess = np.ones(len(studied_quantiles)) / size

    plt.plot(
        random_guess, label="Random Guess", color="grey", ls="-.", linewidth=2
    )
    names_qtiles = name_quantiles(studied_quantiles)
    plt.xticks(range(len(studied_quantiles)), names_qtiles, rotation=45)
    plt.legend(fontsize=12)
    # Add text note to the bottom right corner
    # plt.text(1.1, 1.1, f'1% is {size//100} points\nlast quantile is {round(size*(1-quantiles_pb[parameters["pb"]][-1]/100))} points', ha='right', va='bottom')
    # Adding labels, legend, and gr
    plt.xlabel(
        f'{parameters["pb"]} Quantiles of Distance to Barycenter',
        fontsize=12,
    )

    if save:
        plt.savefig(
            Path(
                "Plots",
                "Reidentification",
                f"{parameters['pb']}{parameters['algo']}_Reid_{name}.eps",
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            Path(
                "Plots",
                "Reidentification",
                f"{parameters['pb']}{parameters['algo']}_Reid_{name}.png",
            ),
            bbox_inches="tight",
        )
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_reidentification_heatmap(
    results, parameters, quantile, FullSpace=False, save=False
):
    from scipy.stats import zscore

    medianprops = dict(linestyle="-.", linewidth=1, color="black")
    if FullSpace:
        iso, name = "DBaryFull", "full"
    else:
        iso, name = "DBaryRed", "red"
    n_quantile = len(quantile)
    split_data = {f"{i_quantile}": [] for i_quantile in range(n_quantile)}
    for seed in range(25):
        list_datasets = split_dataframe_by_quantiles(
            results[str(seed)], column_name=iso, quantile_list=quantile
        )
        for i_data, sub_data in enumerate(list_datasets):
            split_data[str(i_data)].append(sub_data["LC"].to_numpy())
    reid = np.array(
        [
            [
                np.mean(np.array(list(split_data[str(i_quantile)][seed])) == 0)
                for seed in range(25)
            ]
            for i_quantile in range(n_quantile)
        ]
    )

    means = np.mean(reid, axis=1)

    # Use predefined quantiles
    studied_quantiles = quantile

    # Calculate z-scores for the means
    means_zscores = zscore(means)

    # Prepare data for bar plot
    data = {"x": studied_quantiles, "y": means, "zscore": means_zscores}
    df = pd.DataFrame(data)

    # Plot the bar plot with heat gradient based on z-scores
    # cmap = plt.get_cmap("coolwarm")
    # plt.bar(x=np.arange(1,len(studied_quantiles)+1), height=means, color=cmap(df["zscore"]), label="Vulnerability Heatmap")

    plt.bar(
        x=np.arange(1, len(studied_quantiles) + 1),
        height=means,
        color="white",
        edgecolor="black",
        ls=":",
    )

    # Overlay box plots to show variance
    reid_transposed = reid.T  # Transpose for easier plotting with seaborn
    reid_df = pd.DataFrame(reid_transposed, columns=studied_quantiles)

    plt.boxplot(
        [reid_df[col] for col in reid_df.columns],
        showfliers=False,
        widths=0.8,
        medianprops=medianprops,
    )

    # Add mean line to bar plot
    mean_y_barplot = df["y"].mean()
    plt.axhline(
        y=mean_y_barplot,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Average Risk",
    )

    # Customize the bar plot
    plt.xlabel("Quantiles of Distance to Barycenter", fontsize=14)
    plt.ylabel("Reidentification Risk", fontsize=14)
    name_qu = name_quantiles(studied_quantiles)
    plt.xticks(np.arange(1, len(name_qu) + 1), name_qu, rotation=45)

    # Add legend to the bar plot
    plt.legend(loc="upper left", fontsize=14)

    if save:
        plt.savefig(
            Path(
                "Plots",
                "Reidentification",
                f"{parameters['pb']}{parameters['algo']}_BPReidHeatmap_{name}.eps",
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            Path(
                "Plots",
                "Reidentification",
                f"{parameters['pb']}{parameters['algo']}_BPReidHeatmap_{name}.png",
            ),
            bbox_inches="tight",
        )
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_reidentification_heatmap_duo(
    results1,
    results2,
    parameters,
    quantile,
    name1,
    name2,
    FullSpace=False,
    save=False,
):
    medianprops = dict(linestyle="-.", linewidth=1, color="black")
    if FullSpace:
        iso, name = "DBaryFull", "full"
    else:
        iso, name = "DBaryRed", "red"
    n_quantile = len(quantile)
    split_data1 = {f"{i_quantile}": [] for i_quantile in range(n_quantile)}
    split_data2 = {f"{i_quantile}": [] for i_quantile in range(n_quantile)}
    for seed in range(25):
        list_datasets1 = split_dataframe_by_quantiles(
            results1[str(seed)], column_name=iso, quantile_list=quantile
        )
        list_datasets2 = split_dataframe_by_quantiles(
            results2[str(seed)], column_name=iso, quantile_list=quantile
        )
        for i_data, sub_data1 in enumerate(list_datasets1):
            sub_data2 = list_datasets2[i_data]
            split_data1[str(i_data)].append(sub_data1["LC"].to_numpy())
            split_data2[str(i_data)].append(sub_data2["LC"].to_numpy())
    reid1 = np.array(
        [
            [
                np.mean(
                    np.array(list(split_data1[str(i_quantile)][seed])) == 0
                )
                for seed in range(25)
            ]
            for i_quantile in range(n_quantile)
        ]
    )
    reid2 = np.array(
        [
            [
                np.mean(
                    np.array(list(split_data2[str(i_quantile)][seed])) == 0
                )
                for seed in range(25)
            ]
            for i_quantile in range(n_quantile)
        ]
    )

    means1 = np.mean(reid1, axis=1)
    means2 = np.mean(reid2, axis=1)

    # Use predefined quantiles
    studied_quantiles = quantile

    # Calculate z-scores for the means

    # Prepare data for bar plot
    data1 = {"x": studied_quantiles, "y": means1}
    data2 = {"x": studied_quantiles, "y": means2}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    plt.bar(
        x=np.arange(1, len(studied_quantiles) + 1) * 2.5,
        height=means1,
        color="white",
        edgecolor="black",
        ls=":",
        label=name1.title(),
    )
    plt.bar(
        x=np.arange(1, len(studied_quantiles) + 1) * 2.5 + 1,
        height=means2,
        color="lightgrey",
        edgecolor="black",
        ls=":",
        label=name2.title(),
    )

    # Overlay box plots to show variance
    reid_transposed1 = reid1.T  # Transpose for easier plotting with seaborn
    reid_transposed2 = reid2.T
    reid_df1 = pd.DataFrame(reid_transposed1, columns=studied_quantiles)
    reid_df2 = pd.DataFrame(reid_transposed2, columns=studied_quantiles)

    plt.boxplot(
        positions=np.arange(1, len(studied_quantiles) + 1) * 2.5,
        x=[reid_df1[col] for col in reid_df1.columns],
        showfliers=False,
        widths=0.8,
        medianprops=medianprops,
    )
    plt.boxplot(
        positions=np.arange(1, len(studied_quantiles) + 1) * 2.5 + 1,
        x=[reid_df2[col] for col in reid_df2.columns],
        showfliers=False,
        widths=0.8,
        medianprops=medianprops,
    )

    # Add mean line to bar plot
    mean_y_barplot1 = df1["y"].mean()
    mean_y_barplot2 = df2["y"].mean()
    plt.axhline(
        y=mean_y_barplot1,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Average Risk {name1}".title(),
    )
    plt.axhline(
        y=mean_y_barplot2,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Average Risk {name2}".title(),
    )

    # Customize the bar plot
    plt.xlabel("Quantiles of Distance to Barycenter", fontsize=14)
    plt.ylabel("Reidentification Risk", fontsize=14)
    name_qu = name_quantiles(studied_quantiles)
    plt.xticks(np.arange(1, len(name_qu) + 1) * 2.5 + 1, name_qu, rotation=45)

    # Add legend to the bar plot
    plt.legend(loc="upper left", fontsize=14)

    if save:
        plt.savefig(
            Path(
                "Plots",
                "Reidentification",
                f"{parameters['pb']}_ReidHeatmapDouble_{name1+name2+name}.eps",
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            Path(
                "Plots",
                "Reidentification",
                f"{parameters['pb']}_ReidHeatmapDouble_{name1+name2+name}.png",
            ),
            bbox_inches="tight",
        )
    plt.tight_layout()
    plt.show()


def prepare_split_reid(results, quantiles, iso):
    from scipy.stats import zscore

    n_quantile = len(quantiles)
    split_data = {f"{i_quantile}": [] for i_quantile in range(n_quantile)}
    for seed in range(25):
        list_datasets = split_dataframe_by_quantiles(
            results[str(seed)], column_name=iso, quantile_list=quantiles
        )
        for i_data, sub_data in enumerate(list_datasets):
            split_data[str(i_data)].append(sub_data["LC"].to_numpy())
    reid = np.array(
        [
            [
                np.mean(np.array(list(split_data[str(i_quantile)][seed])) == 0)
                for seed in range(25)
            ]
            for i_quantile in range(n_quantile)
        ]
    )

    means = np.mean(reid, axis=1)

    # Calculate z-scores for the means
    means_zscores = zscore(means)

    # Prepare data for bar plot
    data = {"x": quantiles, "y": means, "zscore": means_zscores}
    df = pd.DataFrame(data)
    # Overlay box plots to show variance
    reid_transposed = reid.T  # Transpose for easier plotting with seaborn
    reid_df = pd.DataFrame(reid_transposed, columns=quantiles)
    return df, reid_df


def plot_reidentification_heatmap_versus(
    results, new_results, parameters, FullSpace=False, save=False
):
    # medianprops = dict(linestyle=":", linewidth=1, color="black")
    if FullSpace:
        iso = "DBaryFull"
    else:
        iso = "DBaryRed"

    quantiles = [0, 10, 50, 90]  # quantiles_pb[parameters["pb"]]
    df_van, reid_df = prepare_split_reid(results, quantiles, iso)
    df_vs, reid_vs = prepare_split_reid(new_results, quantiles, iso)
    name_qu = name_quantiles(quantiles)

    groups = quantiles  # list(df_van.keys())
    positions = []
    data1_list = []
    data2_list = []
    for i, group in enumerate(groups):
        positions.extend(
            [i * 3, i * 3 + 1]
        )  # positions for boxplots with gaps
        data1_list.append(reid_df[group])
        data2_list.append(reid_vs[group])

    # Plotting
    fig, ax = plt.subplots()
    ax.boxplot(
        data1_list,
        positions=[p for p in positions if positions.index(p) % 2 == 0],
        widths=0.6,
    )
    ax.boxplot(
        data2_list,
        positions=[p for p in positions if positions.index(p) % 2 == 1],
        widths=0.6,
    )

    # Customizing the x-axis
    ax.set_xticks([i * 3 + 0.5 for i in range(len(groups))])
    ax.set_xticklabels(name_qu)
    plt.text(
        3 * len(quantiles) - 2,
        0.9 * np.max(data1_list + data2_list),
        f'10% is {len(results["0"])//10} rows\n left k is 20-20-20-20\n right k is 5-10-20-40',
        ha="right",
        va="bottom",
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    plt.xlabel("Quantiles of Distance to Barycenter")
    plt.ylabel("Reidentification Risk")
    plt.show()


def compare_reidentification(curves, parameters):
    studied_quantiles = quantiles_pb[parameters["pb"]]

    for curv in curves:
        res = curv[0]["LocalCloaking"]
        reid = np.array(
            [
                [
                    np.mean(np.array(list(res[i_quantile][f"{seed}"])) == 0)
                    for seed in range(25)
                ]
                for i_quantile in res.keys()
            ]
        )

        means = np.mean(reid, axis=1)

        plt.plot(
            range(reid.shape[0]),
            means,
            label=curv[1],
            color=curv[3],
            ls=curv[2],
        )

    plt.title(
        f'Reidentification Metrics Over Quantiles for {len(res["0"].keys())} Attacks - {parameters["pb"]}'
    )

    plt.ylabel("Reidentification")
    plt.legend()

    studied_quantiles = quantiles_pb[parameters["pb"]]

    names_qtiles = name_quantiles(studied_quantiles)
    plt.xticks(range(len(studied_quantiles)), names_qtiles, rotation=45)

    # Add text note to the bottom right corner

    # Adding labels, legend, and gr
    plt.xlabel(f'{parameters["pb"]} Quantiles of Distance to Barycenter')
    plt.tight_layout()
    plt.show()


import matplotlib.patheffects as PathEffects


def overlap_percentage(df, col1, col2, x):
    """
    Calculate the percentage of top x% values in col1 that are also in the top x% values in col2.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.
    x (float): The percentage of the top values to consider.

    Returns:
    float: The percentage of overlap.
    """
    # Number of top x% values
    n_x = int(len(df) * x / 100)

    # Get the indices of the top x% values in col1
    top_x_indices_col1 = df[col1].nlargest(n_x).index

    # Get the indices of the top x% values in col2
    top_x_indices_col2 = df[col2].nlargest(n_x).index

    # Calculate the intersection of these indices
    overlap_indices = top_x_indices_col1.intersection(top_x_indices_col2)

    # Calculate the percentage of overlap
    if n_x != 0:
        overlap_percentage = len(overlap_indices) / n_x * 100
    else:
        overlap_percentage = np.NaN
    return overlap_percentage


def get_curve_arr(arr, x):
    return stats.gaussian_kde(arr)(x)


def Tail_Overlap(df_topo):
    x = np.linspace(0.1, 100, 400)
    data_f = np.zeros(len(x))
    data_r = np.zeros(len(x))
    for i_x, x_ in tqdm(enumerate(x)):
        data_f[i_x] = overlap_percentage(df_topo, "dO", "dA", x_)
        data_r[i_x] = overlap_percentage(df_topo, "do", "da", x_)
    plt.figure(figsize=(6, 5), dpi=80)
    plt.plot(
        x / 100, data_f / 100, color="CornFlowerBlue", label="Original Space"
    )
    plt.plot(x / 100, data_r / 100, color="DarkOrange", label="Latent Space")
    plt.plot([], [], " ", label=f"1% is {len(df_topo)//100} rows")
    plt.legend(fontsize=12)
    plt.xlabel("Percentage of highest Distance to Center", fontsize=15)
    plt.ylabel("Topologie conservation Percentage", fontsize=15)
    plt.show()

    x = np.linspace(0.1, 5, 400)
    data_f = np.zeros(len(x))
    data_r = np.zeros(len(x))
    for i_x, x_ in tqdm(enumerate(x)):
        data_f[i_x] = overlap_percentage(df_topo, "dO", "dA", x_)
        data_r[i_x] = overlap_percentage(df_topo, "do", "da", x_)
    plt.figure(figsize=(6, 5), dpi=80)
    plt.plot(
        x / 100, data_f / 100, color="CornFlowerBlue", label="Original Space"
    )
    plt.plot(x / 100, data_r / 100, color="DarkOrange", label="Latent Space")
    plt.plot([], [], " ", label=f"1% is {len(df_topo)//100} rows")
    plt.legend(fontsize=12)
    plt.xlabel("Percentage of highest Distance to Center", fontsize=15)
    plt.ylabel("Topologie conservation Percentage", fontsize=15)
    plt.show()


def count_common_elements(list1, list2):
    """
    Count the number of common elements between sublists of list1 and list2 and store the results in a matrix.

    Parameters:
    list1 (list of lists): First list containing sublists.
    list2 (list of lists): Second list containing sublists.

    Returns:
    np.ndarray: An n x n matrix with the count of common elements.
    """
    n = len(list1)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            common_elements = set(list1[i]).intersection(list2[j])
            matrix[i, j] = len(common_elements)

    return matrix


def Diffusion_Matrix_multi(
    Data,
    quantiles,
    splitO,
    SplitA,
    name_x="Origine, Quantiles Distance to Barycenter",
    name_y="Avatar, Quantiles Distance to Barycenter",
    save=False,
    title="",
):
    # cols = np.arange(size_split)*step#quantiles_pb[pb]
    name_cols = name_quantiles(quantiles)
    # Compute cross-correlation
    n_quantiles = len(quantiles)
    last_quantiles = np.array(quantiles[1:] + [100])

    values = np.ones((25, n_quantiles, n_quantiles))

    for seed in tqdm(range(25)):
        seed = str(seed)
        list_origine_df = split_dataframe_by_quantiles(
            Data[seed], column_name=splitO, quantile_list=quantiles
        )
        list_avatar_df = split_dataframe_by_quantiles(
            Data[seed], column_name=SplitA, quantile_list=quantiles
        )
        list_origine = [
            np.array(list(sub_df.index)) for sub_df in list_origine_df
        ]
        list_avatar = [
            np.array(list(sub_df.index)) for sub_df in list_avatar_df
        ]
        values[int(seed), :, :] = (
            100
            * (
                100
                * count_common_elements(list_origine, list_avatar)
                / len(Data[seed])
            )
            / (last_quantiles - quantiles).reshape((-1, 1))
        )
    values = np.mean(values, axis=0)

    # Plot cross-correlation matrix
    plt.figure(figsize=(12, 6), dpi=60)
    plt.imshow(
        values, cmap="viridis", interpolation="nearest"
    )  # , vmin=0, vmax=1)
    cbar = plt.colorbar(label="Overlap (%)".title())
    cbar.ax.yaxis.label.set_size(20)
    plt.xlabel(f"{name_x}", fontsize=15)
    plt.xticks(np.arange(n_quantiles), name_cols, rotation=90)
    plt.yticks(np.arange(n_quantiles), name_cols)
    plt.ylabel(f"{name_y}", fontsize=15)
    # plt.yticks(np.arange(values.shape[0]), name_cols, fontsize = 15)
    # plt.xticks(np.arange(values.shape[1]), name_nhit, fontsize = 15)
    plt.grid(False)
    for i in range(n_quantiles):
        for j in range(n_quantiles):
            text = plt.text(
                j,
                i,
                f"{round(values[i, j])}",
                ha="center",
                va="center",
                color="white",
                fontsize=round(160 / n_quantiles),
            )
            # Define the path effect for the outline
            outline_effect = [
                PathEffects.withStroke(linewidth=3, foreground="black")
            ]

            # Apply the path effect to the text
            text.set_path_effects(outline_effect)
    if save:
        plt.savefig(
            Path("Plots", "Topologies", f"{title}.eps"),
            bbox_inches="tight",
        )
        plt.savefig(
            Path("Plots", "Topologies", f"{title}.png"),
            bbox_inches="tight",
        )
    plt.show()


def Conservation_Boxplots(
    pb, Data, quantiles, split, target, name_x, name_y, save=False
):
    # cols = quantiles_pb[pb]#np.arange(size_split)*(100/size_split)#quantiles_pb[pb]
    name_cols = name_quantiles(quantiles)
    # Compute cross-correlation

    list_origine_df = split_dataframe_by_quantiles(
        Data, column_name=split, quantile_list=quantiles
    )
    n_data = len(list_origine_df)
    data_ = [sub_data[target].to_numpy() for sub_data in list_origine_df]

    # Define the positions for the boxplots
    positions = np.arange(n_data)

    # Create the boxplotsf
    fig, ax = plt.subplots()
    ax.boxplot(data_, positions=positions, widths=0.4)
    plt.plot([], [], " ", label=f"1% is {len(Data)//100} rows")
    # Customize the plot
    plt.xticks(positions, name_cols, rotation=90)
    plt.xlabel(name_x, fontsize=15)
    plt.ylabel(name_y, fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    if save:
        1 / 0
        # plt.savefig(Path("Plots_Folder", "PaperPlots", "MIA", f'{pb}{algo}_FilterHM_{target}.eps'), bbox_inches='tight')
    plt.show()


def Conservation_DoubleBoxplots(
    pb, Data, quantiles, split, target, o, a, name_x, name_y, save=False
):
    # cols = quantiles_pb[pb]#np.arange(size_split)*(100/size_split)#quantiles_pb[pb]
    name_cols = name_quantiles(quantiles)
    # Compute cross-correlation

    list_origine_df = split_dataframe_by_quantiles(
        Data, column_name=split, quantile_list=quantiles
    )
    n_data = len(list_origine_df)
    dataO = [sub_data[o + target].to_numpy() for sub_data in list_origine_df]
    dataA = [sub_data[a + target].to_numpy() for sub_data in list_origine_df]

    # Define the positions for the boxplots
    positionsO = np.arange(n_data) * 2 + 1
    positionsA = np.arange(n_data) * 2 + 1.5
    positions = np.arange(n_data) * 2 + 1.25

    # Create the boxplots
    fig, ax = plt.subplots()
    bbplotO = ax.boxplot(
        dataO,
        positions=positionsO,
        widths=0.4,
        label="Origine",
        patch_artist=True,
    )
    bbplotA = ax.boxplot(
        dataA,
        positions=positionsA,
        widths=0.4,
        label="Avatar",
        patch_artist=True,
    )
    plt.plot([], [], " ", label=f"1% is {len(Data)//100} rows")
    plt.legend(fontsize=12)

    for patchO, patchA in zip(bbplotO["boxes"], bbplotA["boxes"]):
        patchO.set_facecolor("CornFlowerBlue")
        patchA.set_facecolor("peachpuff")

    # Customize the plot
    plt.xticks(positions, name_cols, rotation=90)
    plt.xlabel(name_x, fontsize=15)
    plt.ylabel(name_y, fontsize=15)
    plt.legend()
    plt.title("Boxplots of Split Distributions")
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    if save:
        1 / 0
        # plt.savefig(Path("Plots_Folder", "PaperPlots", "MIA", f'{pb}{algo}_FilterHM_{target}.eps'), bbox_inches='tight')
    plt.show()


def compute_saiph(data, processor, model, nf):
    processed = processor.preprocess(data)
    return saiph.transform(processed, model).iloc[:, :nf]


def Isolation_method(df, method_name):
    if method_name == "Distance to Barycenter":
        return df["D_Bary"]
    elif method_name == "Isolation to Neighbors":
        return df["1NN"] + df["3NN"]
    elif method_name == "Both":
        return df["D_Bary"] + df["1NN"]
