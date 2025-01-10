import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np

from copy import copy
import pandas as pd


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


trad_metrics = {
    "SDV": lambda x: ["Column Shapes", "Column Pair Trends"],
    "Bal_Accuracy": lambda x: ["Balanced Accuracy"],
    # "OctoPrivacy":lambda x:["Hidden Rate", "Median Local Cloaking"],
    "Singling_Out_Multi": lambda x: ["Singling Out Multivariate"],
    "Linkability": lambda x: ["Linkability"],
    "Attribute_Risk": sensitive_cols,
    "MIA": lambda x: ["MIA", "MIA_Outlier"],
}

colors_method = {
    "Dataset": "white",
    "Avatar": "white",
    "CTGAN": "white",
    "CompSAIPH": "white",
    "Synthpop": "white",
    "MST": "white",
    "Kanon": "white",
    "MAvatar": "white",
}


def name_translate(list_nm):
    translate_name = {
        "Dataset": "Original\nData   ",
        "CTGAN": "CT-GAN",
        "CompSAIPH": "SAIPH",
        "Kanon": "K-Anony-\nmity",
        "Control": "Control Data   ",
        "Synthpop": "Synth-\npop",
        "MAvatar": "M-\nAvatar",
        "Avatar": "   Avatar",
    }
    list_out = []
    for nm in list_nm:
        if nm in list(translate_name.keys()):
            list_out.append(translate_name[nm])
        else:
            list_out.append(str(nm))
    return list_out


def arr(elem, n=25):
    out = np.array(elem[:n])
    where = np.argwhere(~np.isnan(out))[:, 0]
    return out[where]


target_file = "ThesisPlots"
Metrics_file = "PaperMetrics"
medianprops = dict(linestyle="-.", linewidth=1, color="black")

output = "ThesisMetrics"

size_xticks = 11
size_yticks = 15


list_methods_origine = [
    "Dataset",
    "Avatar",
    "CompSAIPH",
    "MAvatar",
    "CTGAN",
    "Synthpop",
    "MST",
    "Kanon",
]  # , "AvatarResampler200"]


#
for pb_ in [
    "AIDS",
    "WBCD",
    "LAWS",
    "FEWADULT",
    "COMPAS",
    "MEPS",
    "CREDIT",
]:  # dict_loader.keys():
    list_methods = list_methods_origine
    list_metrics = [
        "Balanced Accuracy",
        "Linkability",
        "MIA",
    ] + sensitive_cols(
        pb_
    )  # + ["Singling Out Multivariate"]
    extended_metrics = list_metrics + [
        "Column Pair Trends",
        "Column Shapes",
    ]
    results_dict = {}
    extended_methods = list_methods  # +["Control"]
    for algo in extended_methods:
        results_dict[algo] = {}
        for metric in extended_metrics:
            results_dict[algo][metric] = pickle.load(
                open(
                    Path("Metrics", f"{pb_}_{algo}_{metric}.pkl"),
                    "rb",
                )
            )
    trad_name_metrics = {key: key for key in extended_metrics}
    trad_name_metrics["Linkability"] = "Linkability Risk"
    trad_name_metrics["Singling Out Multivariate"] = "Singling Out Risk"
    trad_name_metrics["MIA"] = "Membership Inference Risk"
    pack_metrics = [trad_metrics[key_](pb_) for key_ in trad_metrics.keys()]
    list_methods_sdv = copy(list_methods)
    methods = [s.replace("_", " ") for s in list_methods]
    methods = name_translate(methods)
    methods_sdv = copy(methods)
    print("SDV", pb_)
    means = {
        method_name: (
            np.mean(arr(results_dict[method_name]["Column Shapes"]))
            + np.mean(arr(results_dict[method_name]["Column Pair Trends"]))
        )
        / 2
        for method_name in list_methods_sdv
    }
    vals = {
        method_name: (
            np.array(arr(results_dict[method_name]["Column Shapes"]))
            + np.array(arr(results_dict[method_name]["Column Pair Trends"]))
        )
        / 2
        for method_name in list_methods_sdv
    }
    plt.figure(figsize=(6, 5), dpi=80)
    plt.bar(
        methods_sdv,
        means.values(),
        color=[colors_method[method] for method in list_methods],
        edgecolor="black",
        ls=":",
    )
    plt.boxplot(
        vals.values(),
        positions=np.arange(len(methods)),
        widths=0.8,
        medianprops=medianprops,
        showfliers=False,
    )
    plt.xticks(
        np.arange(len(methods)),
        methods_sdv,
        rotation=0,
        fontsize=size_xticks,
        ha="center",
    )  # Rotate by 30 degrees and set fontsize to 16

    plt.ylabel("SDV Quality Score", fontsize=size_yticks)
    min_val = np.min(np.array(np.concatenate(list(vals.values())))) * 0.9
    plt.ylim(ymin=min_val)
    plt.savefig(
        Path("Plots", "BoxplotsMetrics", f"{pb_}_0SDV.eps"),
        bbox_inches="tight",
    )
    plt.show()

    for i, metric in enumerate(list_metrics):
        print(metric, pb_)
        means = {
            method_name: np.mean(arr(results_dict[method_name][metric]))
            for method_name in list_methods
        }
        vals = {
            method_name: arr(results_dict[method_name][metric])
            for method_name in list_methods
        }

        plt.figure(figsize=(6, 5), dpi=80)
        plt.bar(
            methods,
            means.values(),
            color=[colors_method[method] for method in list_methods],
            edgecolor="black",
            ls=":",
        )
        plt.boxplot(
            vals.values(),
            # tick_labels=methods,
            positions=np.arange(len(methods)),
            widths=0.8,
            medianprops=medianprops,
            showfliers=False,
        )
        plt.xticks(
            np.arange(len(methods)),
            methods,
            rotation=0,
            fontsize=size_xticks,
            ha="center",
        )  # Rotate by 30 degrees and set fontsize to 16

        min_val = np.min(np.array(np.concatenate(list(vals.values())))) * 0.9
        plt.ylabel(trad_name_metrics[metric].title(), fontsize=size_yticks)
        # if metric == "Balanced Accuracy":
        plt.ylim(ymin=min_val)
        metric_name = metric.replace(" ", "_")
        plt.savefig(
            Path(
                "Plots",
                "BoxplotsMetrics",
                f"{pb_}_{i+1}{metric_name}.eps",
            ),
            bbox_inches="tight",
        )
        plt.show()


colors_method = {
    "Dataset": "white",
    "Avatar": "white",
    "CTGAN": "white",
    "CompSAIPH": "white",
    "Synthpop": "white",
    "MST": "white",
    "MST1": "white",
    "MST2": "white",
    "MST3": "white",
    "Kanon": "white",
    "AvatarResampler200": "white",
    "AvatarResampler50": "white",
    "AvatarResamplerCond": "white",
}


def name_translate(list_nm):
    translate_name = {
        "Dataset": "Original\nData   ",
        "CTGAN": "CT-GAN",
        "CompSAIPH": "SAIPH",
        "Kanon": "K-Anony-\nmity",
        "Control": "Control Data   ",
        "Synthpop": "Synth-\npop",
        "MAvatar": "M-\nAvatar",
        "Avatar": "   Avatar",
    }
    list_out = []
    for nm in list_nm:
        if nm in list(translate_name.keys()):
            list_out.append(translate_name[nm])
        else:
            list_out.append(str(nm))
    return list_out


def arr(elem, n=25):
    out = np.array(elem[:n])
    where = np.argwhere(~np.isnan(out))[:, 0]
    return out[where]


target_file = "PaperPlots"
Metrics_file = "PaperMetrics"
medianprops = dict(linestyle="-.", linewidth=1, color="black")

output = "PublicationMetrics"

size_xticks = 11
size_yticks = 15

list_methods_origine = [
    "Dataset",
    "Avatar",
    "CompSAIPH",
    "MAvatar",
    "CTGAN",
    "Synthpop",
    "MST",
    "Kanon",
]  # , "AvatarResampler200"

# pb_ = "AIDS"
for pb_ in [
    "AIDS",
    "WBCD",
    "LAWS",
    "FEWADULT",
    "COMPAS",
    "MEPS",
    "CREDIT",
]:
    list_methods = list_methods_origine
    # if pb_ == "AIDS":
    #     list_methods = list_methods_origine[:3] +["AvatarResamplerCond"] +list_methods_origine[3:]
    # else:
    #     list_methods = list_methods_origine
    list_metrics = [
        "Balanced Accuracy",
        "Linkability",
        "Singling Out Multivariate",
        "MIA",
    ] + sensitive_cols(pb_)
    if pb_ == "CREDIT":
        list_metrics.remove(
            "Singling Out Multivariate",
        )
    extended_metrics = list_metrics + [
        "Column Pair Trends",
        "Column Shapes",
    ]  # , "MIA_Outlier"] + [risk+" NN" for risk in sensitive_cols(pb_)] + [risk+" NN Outlier" for risk in sensitive_cols(pb_)] + [risk+" Outliers" for risk in sensitive_cols(pb_)]
    results_dict = {}
    extended_methods = list_methods  # +["Control"]
    for algo in extended_methods:
        results_dict[algo] = {}
        for metric in extended_metrics:
            results_dict[algo][metric] = pickle.load(
                open(
                    Path("Metrics", f"{pb_}_{algo}_{metric}.pkl"),
                    "rb",
                )
            )
    trad_name_metrics = {key: key for key in extended_metrics}
    trad_name_metrics["Linkability"] = "Linkability Risk"
    trad_name_metrics["Singling Out Multivariate"] = "Singling Out Risk"
    trad_name_metrics["MIA"] = "Membership Inference Risk"
    pack_metrics = [trad_metrics[key_](pb_) for key_ in trad_metrics.keys()]
    list_methods_sdv = copy(list_methods)
    # list_methods_sdv[0] = "Control"
    methods = [s.replace("_", " ") for s in list_methods]
    methods = name_translate(methods)
    methods_sdv = copy(methods)
    methods_sdv[0] = "Control Data   "

    # Generate LaTeX table summarizing the mean of the boxplots, algorithms in columns, metrics in lines

    # Create mapping from method names to display names
    method_name_mapping = dict(zip(list_methods, methods))
    method_name_mapping_sdv = dict(zip(list_methods_sdv, methods_sdv))

    # Create mapping from metric names to display names
    metric_name_mapping = trad_name_metrics

    # Create list of metrics including 'SDV Quality Score'
    list_metrics_with_sdv = list_metrics + ["SDV Quality Score"]

    # Create DataFrame
    df = pd.DataFrame(
        index=[
            metric_name_mapping.get(metric, metric)
            for metric in list_metrics_with_sdv
        ],
        columns=methods_sdv,
    )

    # Fill in DataFrame for regular metrics
    for metric in list_metrics:
        metric_display_name = metric_name_mapping.get(metric, metric)
        for method in list_methods:
            method_display_name = method_name_mapping[method]
            mean_value = np.mean(arr(results_dict[method][metric]))
            df.loc[metric_display_name, method_display_name] = mean_value

    # Fill in DataFrame for 'SDV Quality Score'
    for method in list_methods_sdv:
        method_display_name = method_name_mapping_sdv[method]
        mean_value = (
            np.mean(arr(results_dict[method]["Column Shapes"]))
            + np.mean(arr(results_dict[method]["Column Pair Trends"]))
        ) / 2
        df.loc["SDV Quality Score", method_display_name] = mean_value

    # Format the DataFrame
    df = df.applymap(lambda x: "{0:.3f}".format(x) if pd.notnull(x) else "")

    # Generate LaTeX code
    latex_code = df.to_latex(index=True)

    # # Save to .tex file
    # output_path = Path("Metrics_Folder", output, f"{pb_}_table.tex")
    # with open(output_path, 'w') as f:
    #     f.write(latex_code)

    # print(f"LaTeX table saved to {output_path}")
