import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import dict_loader

from utils.Evaluator import (
    plot_reidentification_heatmap,
    plot_reidentification_heatmap_duo,
)

pb = "AIDS"

dataframes_dict = pd.compat.pickle_compat.load(
    open(
        Path(
            "Metrics",
            "Reidentification",
            f"{pb}Avatar__Results.pickle",
        ),
        "rb",
    )
)


outliers_dict = {}

for key in dataframes_dict.keys():
    dataframes_dict[key]["ID"] = np.array(dataframes_dict[key].index)
    outliers_dict[key] = dataframes_dict[key].iloc[
        np.argwhere(
            dataframes_dict[key]["DBaryRed"]
            >= np.quantile(dataframes_dict[key]["DBaryRed"], 0.95)
        )[:, 0]
    ]

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes_dict.values(), ignore_index=True)

# Calculate the percentage of rows where 'LC' is 0 for each ID
result = (
    combined_df.groupby("ID")["LC"]
    .apply(lambda x: (x == 0).mean() * 100)  # Calculate percentage
    .reset_index(name="Percentage_LC_0")  # Reset index and name the column
)

Reid = np.array(result["Percentage_LC_0"])


# Combine all dataframes into a single dataframe
combined_df_out = pd.concat(outliers_dict.values(), ignore_index=True)

# Calculate the percentage of rows where 'LC' is 0 for each ID
result_out = (
    combined_df_out.groupby("ID")["LC"]
    .apply(lambda x: (x == 0).mean() * 100)  # Calculate percentage
    .reset_index(name="Percentage_LC_0")  # Reset index and name the column
)

Reid_out = np.array(result_out["Percentage_LC_0"])


def c(arr):
    return np.append(arr, 1)


plt.figure(figsize=(5, 4), dpi=300)

sorted_data = np.sort(Reid)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
plt.plot(
    c(sorted_data / 100), c(cdf), color="CornFlowerBlue", label="Whole Dataset"
)

sorted_data_out = np.sort(Reid_out)
cdf_out = np.arange(1, len(sorted_data_out) + 1) / len(sorted_data_out)
plt.plot(
    c(sorted_data_out / 100),
    c(cdf_out),
    ls=":",
    color="CornFlowerBlue",
    label="5% Outliers",
)

plt.xlabel("Reidentification Risk")
plt.ylabel("CDF")
plt.legend()
plt.grid()
plt.savefig(
    Path("Plots", "Reidentification", f"Reid{pb}.eps"), bbox_inches="tight"
)
plt.savefig(
    Path("Plots", "Reidentification", f"Reid{pb}.png"), bbox_inches="tight"
)
plt.show()


size_small = 12
size_big = 300


def visu(pb):
    params = {
        # "algo": "Avatar",
        "pb": pb,
        "nf": 5,
        "range_seeds": 25,
    }

    plot_reidentification_heatmap_duo(
        pd.compat.pickle_compat.load(
            open(
                Path(
                    "Metrics",
                    "Reidentification",
                    f"{pb}Avatar__Results.pickle",
                ),
                "rb",
            )
        ),
        pd.compat.pickle_compat.load(
            open(
                Path(
                    "Metrics",
                    "Reidentification",
                    f"{pb}CompSAIPH__Results.pickle",
                ),
                "rb",
            )
        ),
        params,
        [0, 10, 20, 30, 50, 90, 97.5],
        "Avatar",
        "SAIPH",
        FullSpace=True,
        save=True,
    )
    plot_reidentification_heatmap_duo(
        pd.compat.pickle_compat.load(
            open(
                Path(
                    "Metrics",
                    "Reidentification",
                    f"{pb}Avatar__Results.pickle",
                ),
                "rb",
            )
        ),
        pd.compat.pickle_compat.load(
            open(
                Path(
                    "Metrics",
                    "Reidentification",
                    f"{pb}CompSAIPH__Results.pickle",
                ),
                "rb",
            )
        ),
        params,
        [0, 10, 20, 30, 50, 90, 97.5],
        "Avatar",
        "SAIPH",
        FullSpace=False,
        save=True,
    )


for pb in dict_loader.keys():
    visu(pb)
