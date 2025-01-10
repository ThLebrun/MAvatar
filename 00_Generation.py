from utils.utils import dict_generator, selection_mia, dict_xp_mia, func_loader
from pathlib import Path
import pickle
import time
import numpy as np

list_datasets = [
    "AIDS",
    "WBCD",
    "FEWADULT",
    "COMPAS",
    "MEPS",
    "CREDIT",
    "LAWS",
]  # ,

import sys

try:
    algo = sys.argv[1]
except:
    algo = "MAvatar"  # "MAvatar", "CTGAN", "Kanon", "MST", 'CompSAIPH'

for pb in [
    "AIDS"
]:  # list_datasets:  # list_datasets:#"LAWS", "WBCD", 'CREDIT' -> CompSAIPH
    if algo in [
        "CompSAIPH",
        "MAvatar",
    ]:  # To cope with SAIPH crashes, one improvement would be to
        i, i_xp = 0, -1
        while i_xp < 1000 and i < 25:
            i_xp += 1
            # try:
            df = func_loader[pb]()
            train_list, control_list = selection_mia(df, i_xp)
            t1 = time.time()
            df = dict_generator[pb][algo](seed_=i_xp, list_ids=train_list)
            t2 = time.time()
            df.to_csv(
                Path(
                    "Data",
                    "DataMIA",
                    "{}MIA_{}_seed{}.csv".format(algo, pb, i),
                ),
                index=False,
            )
            with open(
                Path(
                    "Data",
                    "DataMIA",
                    "{}splitMIA_{}_seed{}.pkl".format(algo, pb, i),
                ),
                "wb",
            ) as pickle_file:
                pickle.dump(
                    {"train": train_list, "test": control_list},
                    pickle_file,
                )
            np.save(
                Path("TimeEval", f"{pb}_{algo}_{i}.npy"), np.array(t2 - t1)
            )
            print(df, f"{pb} {algo} {i} computed")
            i += 1
            # except:
            #     print(f"gen {i_xp} failed")
        df = func_loader[pb]()
        df = dict_generator[pb][algo](seed_=1, list_ids=list())
        df.to_csv(Path("Data", "{}_{}.csv".format(algo, pb)), index=False)
    else:
        for i, i_xp in enumerate(dict_xp_mia[pb]):
            df = func_loader[pb]()
            train_list, _ = selection_mia(df, i_xp)
            t1 = time.time()
            df = dict_generator[pb][algo](seed_=i_xp, list_ids=train_list)
            t2 = time.time()
            np.save(
                Path("TimeEval", f"{pb}_{algo}_{i}.npy"), np.array(t2 - t1)
            )
            df.to_csv(
                Path(
                    "Data",
                    "DataMIA",
                    "{}MIA_{}_seed{}.csv".format(algo, pb, i),
                ),
                index=False,
            )
            print(df, f"{pb} {algo} {i} computed")

        df = func_loader[pb]()
        df = dict_generator[pb][algo](seed_=1, list_ids=list())
        df.to_csv(Path("Data", "{}_{}.csv".format(algo, pb)), index=False)
        print(df, f"{pb} {algo} computed")
