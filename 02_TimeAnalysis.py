# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:54:22 2024

@author: TLebrun
"""
import numpy as np
from pathlib import Path
from utils.data_loader import func_loader, preprocess
import random
import pickle
import pandas as pd


translate_name = {
    "CTGAN": "CT-GAN",
    "CompSAIPH": "SAIPH",
    "COMPSaiph": "SAIPH",
    "Kanon": "K-Anonymity",
    "Synthpop": "Synthpop",
    "MAvatar": "M-Avatar",
    "MST": "MST",
}


def selection_mia(dataset, seed=None, params=None):
    simple = True
    if seed is not None:
        random.seed(seed)  # Set the random seed for reproducibility
    if params is not (None):
        if params["Auxiliary"] == "_AUXout":
            simple = False
            with open(
                Path(
                    "Data",
                    "DataMIA",
                    f'ListMIA_AuxOutlier_{params["pb"]}_{seed}.pickle',
                ),
                "rb",
            ) as f:
                list_member = pickle.load(f)
            set1 = set(list(dataset.index))
            set2 = set(list_member)

            # Elements that are only in list1
            list_non_member = list(set1 - set2)
    if simple:
        list_ids = list(dataset.index)
        random.shuffle(list_ids)
        half_point = len(dataset) // 2
        list_member = list_ids[:half_point]
        list_non_member = list_ids[half_point:]
    return list_member, list_non_member


list_datasets = [
    "AIDS",
    "WBCD",
    "LAWS",
    "FEWADULT",
    "COMPAS",
    "MEPS",
    "CREDIT",
]
list_algos = ["MAvatar", "Synthpop", "Kanon", "CTGAN", "MST", "COMPSaiph"]

nb_parameters = {}
for pb in list_datasets:
    df = func_loader[pb]()
    train_list, _ = selection_mia(df, 0)
    pp_data, _ = preprocess(df.loc[train_list])
    nb_parameters[pb] = pp_data.shape[0] * pp_data.shape[1]


average_points = []
for pb in list_datasets:
    for algo in list_algos:
        points = []
        for seed in range(25):
            if algo == "Synthpop":
                time_value = np.loadtxt(
                    Path(
                        "TimeEval", "{}_{}_{}.txt".format(pb, "Synthpop", seed)
                    )
                )
            else:
                time_value = np.load(
                    Path("TimeEval", "{}_{}_{}.npy".format(pb, algo, seed))
                )
            points.append(time_value)
        average_points.append(
            (
                nb_parameters[pb],
                float(np.mean(points)),
                float(np.min(points)),
                float(np.max(points)),
                algo,
                pb,
            )
        )
df_time = pd.DataFrame(
    np.array(average_points),
    columns=[
        "nb_parameters",
        "mean_time",
        "min_time",
        "max_time",
        "algo",
        "dataset",
    ],
)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming df_time is your DataFrame
# Example structure: ['nb_parameters', 'mean_time', 'min_time', 'max_time', 'algo', 'dataset']

# Group by 'algo' and plot each curve
plt.figure(figsize=(7, 4), dpi=300)

for algo, group in df_time.groupby("algo"):
    # Sort indices based on X
    sorted_indices = np.argsort(np.array(group["nb_parameters"]).astype(float))

    # Sort both X and Y using the sorted indices
    X_sorted = np.array(group["nb_parameters"]).astype(float)[sorted_indices]
    Y_sorted = np.array(group["mean_time"]).astype(float)[sorted_indices]

    plt.plot(
        X_sorted,
        Y_sorted,
        marker="o",
        linestyle="-",
        label=f"{translate_name[algo]}",
    )

# # Set log-log scale
plt.xscale("log")
plt.yscale("log")

# Add labels, title, and legend
plt.xlabel("Number of Parameters (log scale)")
plt.ylabel("Average Computation Time in seconds (log scale)")
# Adjust legend to be outside the plot
plt.legend(
    loc="upper left",  # Legend position within the bounding box
    bbox_to_anchor=(1, 1),  # Position the legend outside the plot
    title="Algorithms",  # Add title to the legend (optional)
)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.savefig(Path("Scalability.eps"), bbox_inches="tight")
plt.show()
