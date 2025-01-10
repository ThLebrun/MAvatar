import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.neighbors import NearestNeighbors
import gower
from tqdm import tqdm
from copy import deepcopy
import saiph

from utils.Generators.SAIPH_Compressor import ToCategoricalProcessor

from utils.data_loader import func_loader, dict_loader
from utils.utils import exp_plan_loader
from utils.Evaluator import distance_to_center

import matplotlib.pyplot as plt

from scipy import stats


def generate_data_overlap(pb, algo="Avatar"):
    if not (algo in ["Avatar", "CompSAIPH"]):
        1 / 0
    results = {}
    data_aux = func_loader[pb]()
    for seed in tqdm(range(25)):
        member_atk, _, synth_atk = exp_plan_loader(pb, algo, seed)

        processor = ToCategoricalProcessor(to_categorical_threshold=20)
        processed_records_aux = processor.preprocess(data_aux)
        processed_members = processor.preprocess(member_atk)

        _, model = saiph.fit_transform(
            processed_records_aux.reset_index(drop=True)
        )

        processed_avatars_mia = processor.preprocess(synth_atk)
        projections_avatars_mia_recomputed = saiph.transform(
            processed_avatars_mia, model
        ).iloc[:, :5]

        pp_data_member = saiph.transform(processed_members, model).iloc[:, :5]

        pp_synth_atk = projections_avatars_mia_recomputed.copy()

        cols = [
            "dO",
            "dA",
            "da",
            "do",
        ]
        result = pd.DataFrame(
            -1 * np.ones((len(member_atk), len(cols))),
            index=member_atk.index,
            columns=cols,
        )

        result["dO"] = distance_to_center(member_atk, "gower")
        result["dA"] = distance_to_center(synth_atk, "gower")
        result["do"] = distance_to_center(pp_data_member, "euclidean")
        result["da"] = distance_to_center(pp_synth_atk, "euclidean")

        results[f"{seed}"] = deepcopy(result)
    pickle.dump(
        results,
        open(Path("Metrics", "Topologies", f"{pb}_{algo}.pickle"), "wb"),
    )
    return results


def get_curve(arr, x):
    kde = stats.gaussian_kde(arr)
    y_vals = kde(x)
    y_vals_percentage = y_vals / y_vals.sum() * 100
    return y_vals_percentage


def distrib_comparaison(
    real, syn, pb, nb_outliers, name_space, name_dist, save=False
):
    plt.figure(figsize=(6, 5), dpi=80)

    data_real = np.array(real)
    data_syn = np.array(syn)
    whole_data = np.concatenate([data_real, data_syn])
    x = np.linspace(0.9 * np.min(whole_data), np.max(whole_data), 200)
    real_curve = get_curve(real, x)
    syn_curve = get_curve(syn, x)
    plt.plot(x, real_curve, color="blue", label="Origine Data")
    plt.plot(x, syn_curve, color="darkorange", label="Avatar Data", ls="--")
    # plt.fill_between(x[x >= threshold], real_curve[x >= threshold], color='blue', alpha=0.3, label = f"{nb_outliers} Origine Outliers")#" = {percentage}%")

    # Get the corresponding values from the second distribution
    plt.xlim(0.9 * np.min(whole_data))
    # plt.scatter(corresponding_values, corresponding_densities, marker='o', color='darkred', label='Originals of Outliers in Avatars')
    plt.xlabel(f"{name_dist} Distance to Barycenter", fontsize=15)
    plt.ylabel("Distribution", fontsize=15)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1f}%")
    )
    plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.6, 1))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save:
        plt.savefig(
            Path(
                "Plots",
                "Topologies",
                f"CDF_{pb}_D_Bary_{name_space}_nofill.eps",
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            Path(
                "Plots",
                "Topologies",
                f"CDF_{pb}_D_Bary_{name_space}_nofill.png",
            ),
            bbox_inches="tight",
        )

    plt.show()


for pb in [
    "AIDS",
    "WBCD",
    "LAWS",
    "COMPAS",
    "FEWADULT",
    "CREDIT",
    "MEPS",
]:
    generate_data_overlap(pb, "CompSAIPH")
    generate_data_overlap(pb, "Avatar")
    Path_name = Path("Metrics", "Topologies", f"{pb}_Avatar.pickle")
    data_topo = pd.compat.pickle_compat.load(open(Path_name, "rb"))

    seed = 1
    do = data_topo[list(data_topo.keys())[seed]]["do"]
    da = data_topo[list(data_topo.keys())[seed]]["da"]
    dO = data_topo[list(data_topo.keys())[seed]]["dO"]
    dA = data_topo[list(data_topo.keys())[seed]]["dA"]
    distrib_comparaison(do, da, pb, 20, "Reduced", "Euclidean", True)

    distrib_comparaison(dO, dA, pb, 20, "Full", "Gower", True)
