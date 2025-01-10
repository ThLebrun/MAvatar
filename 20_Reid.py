from pathlib import Path
from utils.Evaluator import (
    plot_reidentification,
    Reid_Attack,
    plot_reidentification_heatmap,
    dict_loader,
)

import pandas as pd


import pickle

ploting = True
Full_Data = False
computing = True

FD = ""
if Full_Data:
    FD = "FD"
# "Avatar",
for (
    pb
) in (
    dict_loader.keys()
):  # dict_loader.keys():#["FEWADULT"]:#["AIDS", "WBCD", "MEPS", "COMPAS", "CREDIT", "LAWS"]:
    for algo in [
        "CompSAIPH",
        "Avatar",
    ]:  # ""Avatar", "Avatar_ncp2_k20", "Avatar_ncp5_k20", "Avatar_ncp10_k20", "Avatar_ncp5_k5", "Avatar_ncp5_k40"]:
        params = {
            "algo": algo,
            "pb": pb,
            "nf": 5,
            "range_seeds": 25,
        }
        if computing:
            results = Reid_Attack(params)
            pickle.dump(
                results,
                open(
                    Path(
                        "Metrics",
                        "Reidentification",
                        f"{pb}{algo}_{FD}_Results.pickle",
                    ),
                    "wb",
                ),
            )
        if ploting:
            results = pd.compat.pickle_compat.load(
                open(
                    Path(
                        "Metrics",
                        "Reidentification",
                        f"{pb}{algo}_{FD}_Results.pickle",
                    ),
                    "rb",
                )
            )
            plot_reidentification(results, params, FullSpace=True, save=True)
            plot_reidentification(results, params, FullSpace=False, save=True)
            # plot_reidentification_heatmap(results, params, FullSpace=True, save=True)
            plot_reidentification_heatmap(
                results,
                params,
                [0, 10, 20, 30, 50, 90, 97.5],
                FullSpace=True,
                save=True,
            )
            plot_reidentification_heatmap(
                results,
                params,
                [0, 10, 20, 30, 50, 90, 97.5],
                FullSpace=False,
                save=True,
            )
#     print(pb)

# for pb in ["AIDS", "WBCD", "MEPS", "COMPAS", "CREDIT", "LAWS"]:#, "WBCD", "MEPS", "COMPAS", "CREDIT", "LAWS"]:
#     params = create_parameters((pb, "None"), "Avatar")
#     results = pickle.load(open(Path("Results_Folder", f"ResultsReid{FD}", f"{pb}NoneAvatar_Results.pickle"), "rb"))
#     results_versus = pickle.load(open(Path("Results_Folder", f"ResultsReid{FD}", f"{pb}NoneAvatarKvar_wide_Results.pickle"), "rb"))
#     plot_reidentification_heatmap_versus(results, results_versus, params, FullSpace=False, save=True)
