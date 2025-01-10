from utils.Evaluator import compute_metric
from utils.utils import dict_loader
from tqdm import tqdm
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

list_datasets = dict_loader.keys()
list_algos = [
    "Synthpop",
    "Control",
    "MST",
    "Kanon",
    "CompSAIPH",
    "CTGAN",
    "Avatar",
    "Dataset",
    "MAvatar",
]  # ["MST"]#"Avatar", "CTGAN", "Dataset", "Synthpop", "Kanon"]#"Control"#"CTGAN", "Reproduction_Avatar", "Reproduction_Avatar_2", "Reproduction_Avatar_3", "Dataset", "Synthpop", "Kanon"]#["CTGAN", "Reproduction_Avatar", "Reproduction_Avatar_2"]#, "CTGAN", "Reproduction_Avatar", "Reproduction_Avatar_2", "Kanon", "Synthpop"]
# list_algos = [
#     "Synthpop",
#     "Control",
#     "MST",
#     "Kanon",
#     "CompSAIPH",
#     "CTGAN",
#     "Control",
#     "Dataset",
#     "MAvatar",
# ]
list_metrics = [
    "SDV",
    "Bal_Accuracy",
    "Linkability",
    "Attribute_Risk",
    "Singling_Out_Multi",
    "MIA",
]  # ["SDV", "Bal_Accuracy", "Linkability", "Attribute_Risk", "Singling_Out_Multi", "MIA"]#["SDV", "Bal_Accuracy", "Linkability", "Attribute_Risk", "Singling_Out_Multi"]# ["MIA"]#["Bal_Accuracy"]#[]# "Singling_Out_Multi"["SDV", "Bal_Accuracy", "Linkability", "Attribute_Risk", "Singling_Out_Multi"]#, "Bal_Accuracy", "OctoPrivacy", "Singling_Out_Multi", "Linkability", "Attribute_Risk"]

exp_plan = [
    (pb, algo, metric)
    for metric in [
        # "SDV",
        # "Bal_Accuracy",
        # "Linkability",
        # "Attribute_Risk",
        "Singling_Out_Multi",
        # "MIA",
    ]
    for pb in ["FEWADULT"]  # list_datasets
    for algo in ["MAvatar"]  # list_algos
]

for pb, algo, metric in tqdm(exp_plan):  # list_datasets:
    compute_metric(pb, metric, algo)

# compute_metric("AIDS", "MIA", "Avatar")
