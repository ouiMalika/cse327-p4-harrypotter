import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Calculate mean and median nodes explored from a CSV file.")
parser.add_argument("input_file", help="Path to the input CSV file")
parser.add_argument("mode", choices=["std", "ming"], help="Specify whether to analyze 'std' or 'ming' data")
args = parser.parse_args()

df = pd.read_csv(args.input_file)

column_name = f"{args.mode} nodes explored"

mean_nodes = df[column_name].mean()
median_nodes = df[column_name].median()

print(f"Mean Nodes Explored ({args.mode}): {mean_nodes}")
print(f"Median Nodes Explored ({args.mode}): {median_nodes}")
