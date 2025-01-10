import pickle
from pathlib import Path
import numpy as np
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


def name_translate_tabular(list_nm):
    translate_name = {
        "Avatar": "\\avatar",
        "CompSAIPH": "\\saiph",
        "MAvatar": "\\avatarModel",
        "CTGAN": "\\ctgan",
        "Synthpop": "\\synthpop",
        "MST": "\\mst",
        "Kanon": "\\kanon",
        "Dataset": "Orig. Data",
    }
    list_out = []
    for nm in list_nm:
        if nm in translate_name:
            list_out.append(translate_name[nm])
        else:
            list_out.append(str(nm))
    return list_out


def arr(elem, n=25):
    out = np.array(elem[:n])
    where = np.argwhere(~np.isnan(out))[:, 0]
    return out[where]


medianprops = dict(linestyle="-.", linewidth=1, color="black")

full_table = ""
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
]

# Mapping for metric display names
metric_name_mapping = {
    "Balanced Accuracy": "Task Accuracy",
    "SDV Quality Score": "SDV Quality Score",
    "Linkability": "Linkability Risk",
    "Singling Out Multivariate": "Singling Out Risk",
    "MIA": "Membership Inference Risk",
}

metric_name_mapping_tabular = {
    "Balanced Accuracy": "Task Acc.",
    "SDV Quality Score": "SDV Score",
    "Linkability": "Linkability",
    "Singling Out Multivariate": "Singling Out",
    "MIA": "MIA",
}

# For the AIA Risks, we need to include sensitive columns
for pb_ in [
    "AIDS",
    "WBCD",
    "LAWS",
    "FEWADULT",
    "COMPAS",
    "MEPS",
    "CREDIT",
]:
    print(f"Processing dataset: {pb_}")
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
    extended_metrics = list_metrics + ["Column Pair Trends", "Column Shapes"]
    results_dict = {}
    for algo in list_methods_origine:
        results_dict[algo] = {}
        for metric in extended_metrics:
            results_dict[algo][metric] = pickle.load(
                open(
                    Path("Metrics", f"{pb_}_{algo}_{metric}.pkl"),
                    "rb",
                )
            )

    methods_tabular = name_translate_tabular(list_methods_origine)

    # Initialize DataFrame rows
    df_rows = []

    # Collect Task Acc. and SDV Score
    main_metrics = ["Balanced Accuracy", "SDV Quality Score"]
    for metric in main_metrics:
        metric_display_name = metric_name_mapping_tabular.get(metric, metric)
        metric_values = {}
        for method in list_methods_origine:
            if metric == "SDV Quality Score":
                mean_value = (
                    np.mean(arr(results_dict[method]["Column Shapes"]))
                    + np.mean(arr(results_dict[method]["Column Pair Trends"]))
                ) / 2
            else:
                mean_value = np.mean(arr(results_dict[method][metric]))
            metric_values[method] = mean_value

        # Emphasize highest values (within 1% of the max value)
        max_value = max(np.array(list(metric_values.values()))[1:])
        threshold = 0.9 * max_value
        formatted_values = {}
        for method, value in metric_values.items():
            if value >= threshold and method != "Dataset":
                formatted_value = f"\\textbf{{{value:.3f}}}"
            else:
                formatted_value = f"{value:.3f}"
            formatted_values[method] = formatted_value

        row = [metric_display_name] + [
            formatted_values[method] for method in list_methods_origine
        ]
        df_rows.append(row)

    # Add a midrule
    df_rows.append(["\\midrule"] + [""] * len(list_methods_origine))

    # Collect Privacy Metrics
    privacy_metrics = ["Linkability", "Singling Out Multivariate", "MIA"]
    for metric in privacy_metrics:
        # Check if the metric exists (e.g., 'Singling Out Multivariate' may not be in 'CREDIT')
        if metric not in list_metrics:
            continue
        metric_display_name = metric_name_mapping_tabular.get(metric, metric)
        metric_values = {}
        for method in list_methods_origine:
            mean_value = np.mean(arr(results_dict[method][metric]))
            metric_values[method] = mean_value

        # Emphasize lowest values (within 1% above the min value)
        min_value = min(np.array(list(metric_values.values()))[1:])
        threshold = 1.1 * min_value
        formatted_values = {}
        for method, value in metric_values.items():
            if value <= threshold and method != "Dataset":
                formatted_value = f"\\textbf{{{value:.3f}}}"
            else:
                formatted_value = f"{value:.3f}"
            formatted_values[method] = formatted_value

        row = [metric_display_name] + [
            formatted_values[method] for method in list_methods_origine
        ]
        df_rows.append(row)

    # Add a midrule and 'AIA Risks' header
    df_rows.append(["\\midrule"] + [""] * len(list_methods_origine))
    df_rows.append(["\\textbf{AIA Risks}"] + [""] * len(list_methods_origine))

    # Collect AIA Risks
    for sensitive_col in sensitive_cols(pb_):
        metric_display_name = sensitive_col
        metric_display_name = metric_display_name.split("Inference Risk")[0][
            :-1
        ].title()
        metric_values = {}
        for method in list_methods_origine:
            mean_value = np.mean(arr(results_dict[method][sensitive_col]))
            metric_values[method] = mean_value

        # Emphasize lowest values
        min_value = min(np.array(list(metric_values.values()))[1:])
        threshold = 1.1 * min_value
        formatted_values = {}
        for method, value in metric_values.items():
            if value <= threshold:
                formatted_value = f"\\textbf{{{value:.3f}}}"
            else:
                formatted_value = f"{value:.3f}"
            formatted_values[method] = formatted_value

        row = [metric_display_name] + [
            formatted_values[method] for method in list_methods_origine
        ]
        df_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(df_rows)
    # Set column names
    df.columns = [""] + methods_tabular

    # Convert DataFrame to LaTeX table
    latex_table = df.to_latex(index=False, header=True, escape=False)

    # Add LaTeX table formatting
    num_cols = len(list_methods_origine) + 1  # plus one for the index column
    table_header = (
        r"""\begin{table}[h!]
    \centering
    \footnotesize
    \begin{tabular}{l"""
        + "l" * (num_cols - 1)
        + r"""}
    \toprule
"""
    )
    table_footer = (
        r"""\bottomrule
\end{tabular}

%\vspace{2mm}
\caption{"""
        + f"{pb_} - Utility and privacy metrics comparison between the different baselines."
        + r"""}
\label{"""
        + f"{pb_}ResultsMetrics"
        + r"""}
%\vspace{-8mm}
\end{table}
"""
    )

    # Combine parts
    latex_code = (
        table_header
        + latex_table.split("\\toprule")[1].split("\\bottomrule")[0]
        + table_footer
    )
    full_table += (
        latex_code
        + r"""
    
    
    
    """
    )
    # Save to .tex file
    output_path = Path("LatexTabulars", f"{pb_}_table.tex")
    with open(output_path, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to {output_path}")

    # Plotting the figures
    # Plot SDV Quality Score
    means_sdv = {
        method_name: (
            np.mean(arr(results_dict[method_name]["Column Shapes"]))
            + np.mean(arr(results_dict[method_name]["Column Pair Trends"]))
        )
        / 2
        for method_name in list_methods_origine
    }
    vals_sdv = {
        method_name: (
            np.array(arr(results_dict[method_name]["Column Shapes"]))
            + np.array(arr(results_dict[method_name]["Column Pair Trends"]))
        )
        / 2
        for method_name in list_methods_origine
    }


# Save to .tex file
output_path = Path("Metrics", "SyntheticMetrics_table.tex")
with open(output_path, "w") as f:
    f.write(full_table)

print(f"LaTeX table saved to {output_path}")


from pathlib import Path
import numpy as np
import pandas as pd

import re

# LaTeX code as a string


def latex_to_df(latex_code):
    # Extract table content between \begin{tabular} and \end{tabular}
    pattern = r"\\begin\{tabular\}\{[^\}]*\}(.*?)\\end\{tabular\}"
    match = re.search(pattern, latex_code, re.DOTALL)
    if match:
        table_content = match.group(1)
    else:
        table_content = ""

    # Split the content into lines
    lines = table_content.strip().split("\n")

    # Remove lines that are \toprule, \midrule, \bottomrule, or empty
    data_lines = []
    for line in lines:
        line = line.strip()
        if (
            line.startswith("\\toprule")
            or line.startswith("\\midrule")
            or line.startswith("\\bottomrule")
        ):
            continue
        if not line:
            continue
        data_lines.append(line)

    # Split lines into cells
    data = []
    for line in data_lines:
        line = line.rstrip("\\\\").strip()
        cells = [cell.strip() for cell in line.split("&")]
        data.append(cells)

    # Define macro replacements
    macro_mapping = {
        "\\avatar": "AVATAR",
        "\\saiph": "SAIPH",
        "\\avatarModel": "AVATAR Model",
        "\\ctgan": "CTGAN",
        "\\synthpop": "Synthpop",
        "\\mst": "MST",
        "\\kanon": "K-anonymization",
    }

    # Function to clean and replace LaTeX elements
    def clean_cell(cell):
        for macro, name in macro_mapping.items():
            cell = cell.replace(macro, name)
        cell = re.sub(r"\\textbf\{(.*?)\}", r"\1", cell)
        cell = re.sub(r"\\[a-zA-Z]+\s*", "", cell)
        cell = cell.replace("{", "").replace("}", "").strip()
        return cell

    # Clean the data
    data_cleaned = []
    for row in data:
        row_cleaned = [clean_cell(cell) for cell in row]
        data_cleaned.append(row_cleaned)

    # Extract headers
    headers = data_cleaned[0]
    headers[0] = "Metric"  # Rename the first empty header to 'Metric'

    # Process data rows, skipping empty rows and subheaders
    data_rows = []
    for row in data_cleaned[1:]:
        if row[0] and "Risks" not in row[0]:
            data_rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data_rows, columns=headers)

    # Set 'Metric' as the index
    df.set_index("Metric", inplace=True)

    # Convert numeric values to floats
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


perfs_data = {}

for pb in [
    "AIDS",
    "WBCD",
    "LAWS",
    "FEWADULT",
    "COMPAS",
    "MEPS",
    "CREDIT",
]:
    file_path = Path("LatexTabulars", f"{pb}_table.tex")

    # Read the content of the .tex file
    with open(file_path, "r", encoding="utf-8") as file:
        latex_code = file.read()
    perfs_data[pb] = latex_to_df(latex_code)
list_metrics = [
    "Task Acc.",
    "SDV Score",
    "Linkability",
    "Singling Out",
    "MIA",
    "Gender",
    "Race",
]
output_metric = {}
for metric in list_metrics:
    output_metric[metric] = list()
    for pb in [
        "AIDS",
        "WBCD",
        "LAWS",
        "FEWADULT",
        "COMPAS",
        "MEPS",
        "CREDIT",
    ]:
        if metric in list(perfs_data[pb].index):
            output_metric[metric].append(perfs_data[pb].loc[metric].to_numpy())

df_mean = pd.DataFrame(columns=perfs_data[pb].columns)
df_median = pd.DataFrame(columns=perfs_data[pb].columns)
df_min = pd.DataFrame(columns=perfs_data[pb].columns)
df_max = pd.DataFrame(columns=perfs_data[pb].columns)

for metric in list_metrics:
    df_mean.loc[metric] = np.mean(np.array(output_metric[metric]), axis=0)
    df_median.loc[metric] = np.median(np.array(output_metric[metric]), axis=0)
    df_min.loc[metric] = np.min(np.array(output_metric[metric]), axis=0)
    df_max.loc[metric] = np.max(np.array(output_metric[metric]), axis=0)


import numpy as np
import pandas as pd

# Assuming df_min and df_max are already defined as per your code
# Create a new DataFrame to hold the "min - max" strings
df_min_max = pd.DataFrame(index=df_min.index, columns=df_min.columns)

# Populate df_min_max with "min - max" formatted strings
for metric in df_min.index:
    for col in df_min.columns:
        min_val = df_min.loc[metric, col]
        max_val = df_max.loc[metric, col]
        # Format the values to three decimal places
        df_min_max.loc[metric, col] = f"{min_val:.3f} - {max_val:.3f}"

# Optional: Rename the index name if needed
df_min_max.index.name = "Metric"

# Generate the LaTeX code for the table
latex_table = df_min_max.to_latex(
    column_format="l" + "r" * len(df_min_max.columns),
    escape=False,
    index=True,
    bold_rows=True,
    caption="Min-Max Values of Metrics Across Datasets",
    label="tab:min_max_metrics",
    header=True,
    na_rep="--",
    float_format="%.2f",
    # hrules=True
)

# Save the LaTeX code to a file
with open("min_max_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

# Print the LaTeX code (optional)
print(latex_table)

latex_table = df_median.to_latex(
    column_format="l" + "l" * len(df_median.columns),
    escape=False,
    index=True,
    bold_rows=True,
    caption="Median Values of Metrics Across Datasets",
    label="tab:median_metrics",
    header=True,
    na_rep="--",
    float_format="%.3f",
    # hrules=True
)

# Save the LaTeX code to a file
with open("LatexTabulars", "median_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

# Print the LaTeX code (optional)
print(latex_table)
