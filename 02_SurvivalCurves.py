import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from UtilsScripts.data_loader import func_loader

# Specify the file path


def split_monotonic(lst):
    if not lst:
        return []

    result = []
    sublist = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] < lst[i - 1]:  # Check for an abrupt change
            result.append(sublist)
            sublist = [lst[i]]
        else:
            sublist.append(lst[i])

    result.append(sublist)  # Add the last sublist
    return result


def split_surv(initial_split, surv_data):
    surv_data = np.array(surv_data)
    init = 0
    out = []
    for sub_list in initial_split:
        out.append(surv_data[init : init + len(sub_list)])
        init += len(sub_list)
    return out


def get_surv_curves(algo, where):
    # Open and read the JSON file
    file_path = Path("Data", "AIDS_Survival", algo + ".json")
    with open(file_path, "r") as file:
        data_raw = json.load(file)
    split_time = split_monotonic(data_raw["time"])
    data = {
        "time": split_time,
        "surv": split_surv(split_time, data_raw["surv"]),
    }

    time = [0] + data["time"][where] + [500]
    surv = np.zeros(len(time * 2))
    data_surv = [1] + data["surv"][where] + [data["surv"][where][-1]]
    surv[: len(data_surv)] = data["surv"][where]
    surv = surv[: len(time)]
    return [time, surv]


def plot_surv(curves, name, save=True, legend_bool=True):
    if legend_bool:
        plt.figure(figsize=(7, 5), dpi=80)
        legend_name = ""
    else:
        plt.figure(figsize=(7, 5), dpi=80)
        legend_name = "_nolegend"

    for curve in curves:
        plt.plot(
            curve[0][0],
            curve[0][1],
            color=curve[1],
            ls=curve[2],
            label=curve[3],
        )
    if legend_bool:
        plt.legend(loc="lower left", fontsize=12)  # bbox_to_anchor=(1.05, 1),
    plt.ylabel("Zidovadine - Survival rate", fontsize=15)
    plt.xlabel("Time in week", fontsize=15)
    plt.xlim(0, 200)
    plt.ylim(0.495, 1.01)
    if save:
        plt.savefig(
            Path(
                "Plots_Folder",
                "PaperPlots",
                "AIDS_Survival",
                f"AIDS_Survival_{name}{legend_name}.eps",
            ),
            bbox_inches="tight",
        )
    plt.show()


def plot_surv2(curves, name, save=True, legend_bool=True):
    if legend_bool:
        plt.figure(figsize=(10, 6), dpi=300)
        legend_name = ""
    else:
        plt.figure(figsize=(10, 6), dpi=300)
        legend_name = "_nolegend"

    for curve in curves:
        plt.plot(
            curve[0][0],
            curve[0][1],
            color=curve[1],
            ls=curve[2],
            label=curve[3],
        )
    if legend_bool:
        plt.legend(loc="lower left", fontsize=12)  # bbox_to_anchor=(1.05, 1),
    plt.ylabel("Zidovadine - Survival rate", fontsize=15)
    plt.xlabel("Time in week", fontsize=15)
    plt.xlim(0, 200)
    plt.ylim(0.495, 1.01)
    if save:
        plt.savefig(
            Path(f"WISE_AIDS_Survival_{name}{legend_name}.eps"),
            bbox_inches="tight",
        )
    plt.show()


curves_surv = [
    (get_surv_curves("Avatar", 0), "Black", "-", "Original Data"),
    (get_surv_curves("Avatar", 1), "DarkOrange", "--", "Avatar"),
    (get_surv_curves("CT-GAN", 1), "DarkGreen", ":", "CT-GAN"),
    (get_surv_curves("Synthpop", 1), "DarkRed", "-.", "Synthpop"),
    (
        get_surv_curves("MST", 1),
        "CornFlowerBlue",
        (0, (3, 1, 1, 1, 1, 1)),
        "MST",
    ),
    (
        get_surv_curves("Kanon", 1),
        "purple",
        (0, (1, 1, 1, 1, 1, 1)),
        "K-Anonymity",
    ),
    (
        get_surv_curves("ResamplerCond", 1),
        "blue",
        (0, (4, 4, 0, 2, 1, 1)),
        "M-Avatar",
    ),
]


curves_surv_comp = [
    (get_surv_curves("Avatar", 0), "Black", "-", "Original Data"),
    (get_surv_curves("CompSAIPH2", 1), "DarkOrange", "--", "SAIPH2"),
    (get_surv_curves("CompSAIPH3", 1), "DarkGreen", "-.", "SAIPH3"),
    (get_surv_curves("CompSAIPH5", 1), "DarkRed", ":", "SAIPH5"),
    (
        get_surv_curves("CompSAIPH10", 1),
        "CornFlowerBlue",
        (0, (3, 1, 1, 1, 1, 1)),
        "SAIPH10",
    ),
    (
        get_surv_curves("CompSAIPH20", 1),
        "purple",
        (0, (1, 1, 1, 1, 1, 1)),
        "SAIPH20",
    ),
]


curves_surv_ncp = [
    (get_surv_curves("Avatar", 0), "Black", "-", "Original Data"),
    (get_surv_curves("Avatar", 1), "DarkOrange", "--", "Avatar"),
    (get_surv_curves("Avatar5", 1), "DarkOrange", "-.", "Avatar5"),
    (get_surv_curves("Avatar10", 1), "DarkGreen", "-.", "Avatar10"),
    (get_surv_curves("CompSAIPH5", 1), "DarkRed", ":", "SAIPH5"),
    (
        get_surv_curves("CompSAIPH10", 1),
        "CornFlowerBlue",
        (0, (3, 1, 1, 1, 1, 1)),
        "SAIPH10",
    ),
]

curves_surv_resample = [
    (get_surv_curves("Avatar", 0), "Black", "-", "Original Data"),
    (
        get_surv_curves("Avatar", 1),
        "DarkOrange",
        "--",
        "Avatar",
    ),
    (get_surv_curves("Resampler", 1), "blue", "-.", "Avatar Model 1"),
    (get_surv_curves("Resampler_V2", 1), "green", "-.", "Avatar Model 2"),
    (get_surv_curves("Resampler_V3", 1), "red", "-.", "Avatar Model 3"),
    (
        get_surv_curves("Synthpop", 1),
        "CornFlowerBlue",
        (0, (3, 1, 1, 1, 1, 1)),
        "Synthpop",
    ),
]


curves_surv_resample2 = [
    (get_surv_curves("Avatar", 0), "Black", "-", "Original Data"),
    (
        get_surv_curves("Avatar", 1),
        "DarkOrange",
        "--",
        "Avatar",
    ),
    (
        get_surv_curves("ResamplerCond", 1),
        "green",
        "-.",
        "Avatar Model\nConditional",
    ),
    (
        get_surv_curves("Synthpop", 1),
        "CornFlowerBlue",
        (0, (3, 1, 1, 1, 1, 1)),
        "Synthpop",
    ),
]

save_bool = False
plot_surv(curves_surv, "Baselines", save_bool, True)
plot_surv2(curves_surv, "Baselines", save_bool, True)
plot_surv(curves_surv, "Baselines", save_bool, False)
plot_surv(curves_surv_comp, "Compression", save_bool, True)
plot_surv(curves_surv_comp, "Compression", save_bool, False)
plot_surv(curves_surv_ncp, "AvatarCompression", save_bool, True)
plot_surv(curves_surv_ncp, "AvatarCompression", save_bool, False)
