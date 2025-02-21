import pandas as pd
import re

# Load the Excel file
file_path = '/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/Results.xlsx'  # Replace with your actual file path
excel_data = pd.ExcelFile(file_path)

# Load the specific tab
sheet_name = 'BM1_single_step'  # Update this if the sheet name differs
data_df = excel_data.parse(sheet_name)

# Specify the columns to use
columns_to_keep = [
    "Original task name", "AVG", "AVG.1", "AVG.2", "AVG.3", "AVG.4",
    "Modified task name", "AVG.5", "AVG.6", "AVG.7", "AVG.8", "AVG.9",
    "Delta Transformer (ori - modified)", "Delta RNN (ori - modified)",
    "Delta Vilt (ori - modified)", "Delta Openvla (ori - modified)",
    "Delta MaIL (ori - modified)"
]

# Filter only the relevant columns
selected_columns_df = data_df[columns_to_keep].iloc[:44]  # Use rows 1 to 44

# Prepare LaTeX table rows
restructured_data = []

for _, row in selected_columns_df.iterrows():
    try:
        # Extract meaningful task names by removing 'KITCHEN_SCENE**_'
        original_task_name = re.sub(r"KITCHEN_SCENE\d+_", "", str(row["Original task name"])).replace("_", " ")
        modified_task_name = re.sub(r"KITCHEN_SCENE\d+_", "", str(row["Modified task name"])).replace("_", " ")

        # Row 1: Original task name
        original_row = [
            original_task_name,
            row["AVG"], row["AVG.1"], row["AVG.2"], row["AVG.3"], row["AVG.4"]
        ]

        # Row 2: Modified task name
        modified_row = [
            modified_task_name,
            row["AVG.5"], row["AVG.6"], row["AVG.7"], row["AVG.8"], row["AVG.9"]
        ]

        # Row 3: PD (Delta values)
        delta_row = [
            "PD",
            row["Delta Transformer (ori - modified)"],
            row["Delta RNN (ori - modified)"],
            row["Delta Vilt (ori - modified)"],
            row["Delta Openvla (ori - modified)"],
            row["Delta MaIL (ori - modified)"]
        ]

        # Append the three rows as a set, without adding an empty row
        restructured_data.extend([original_row, modified_row, delta_row, ["\\hline\\hline"]])
    except Exception as e:
        print(f"Error processing row: {e}")

# Remove the last separator (to avoid an extra line at the end of the table)
if restructured_data[-1] == ["\\hline\\hline"]:
    restructured_data.pop()

# Create a DataFrame for the LaTeX table
restructured_df = pd.DataFrame(restructured_data, columns=["Task", "BC-RESNET-RNN", "BC-RESNET-T", "BC-VIT-T", "MaIL", "OpenVLA"])

# Generate the LaTeX table
latex_table = restructured_df.to_latex(
    index=False,
    header=True,
    column_format="@{}l@{}c@{}c@{}c@{}c@{}c@{}",  # Reduce column spacing
    escape=False
)

# Correct LaTeX table formatting
table_rows = latex_table.splitlines()
header = table_rows[0].replace("\\", "") + " \\\\"
body = "\n".join(
    row if "\\hline\\hline" not in row else "\\hline\\hline" for row in table_rows[2:-1]
)

# Ensure proper placement of \midrule and formatting
formatted_latex_table = (
    "\\begin{table*}[ht]\n"
    "\\centering\n"
    "\\caption{Results for BM-1 (Rows 1-44)}\n"
    "\\label{tab:restructured_table}\n"
    "\\scriptsize\n"
    "\\begin{tabular}{@{}l@{}c@{}c@{}c@{}c@{}c@{}}\n"
    "\\toprule\n"
    + header + "\n"
    "\\midrule\n"
    + body + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table*}"
)

# Output the LaTeX table
print(formatted_latex_table)
