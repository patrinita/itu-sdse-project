import subprocess
import pandas as pd
import warnings
import datetime
import json
import os

def prepare_data_and_artifacts(raw_data_path="/app/raw/raw_data.csv"):
    """
    Creates directories, loads data, applies date filtering, 
    and saves date limits for artifacts.
    """
    # Creates necessary directories
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    print("Created artifacts and mlruns directories")

    # Configures warnings and pandas display
    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format', lambda x: "%.3f" % x)

    # Pulls latest data using DVC
    # subprocess.run(["dvc", "pull"])
    # print("DVC pull completed")

    # if os.path.exists(".dvc"):
    #     subprocess.run(["dvc", "pull"], check=True)
    # else:
    #     print("Skipping DVC pull (not in DVC repo)")
    dvc_dir = "code/.dvc"
    if os.path.isdir(dvc_dir):
        subprocess.run(["dvc", "pull"], cwd="code", check=True)
        print("DVC pull completed")
    else:
        print("Skipping DVC pull (code/.dvc not found)")


    # Loads training data
    print("Loading training data")
    data = pd.read_csv(raw_data_path)

    print("Total rows:", data.count())
    print(data.head(5))

    # Date filtering
    max_date = "2024-01-31"
    min_date = "2024-01-01"

    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()

    # Filters data by date
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    # Saves date limits
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    print("Date limits saved to artifacts/date_limits.json")
    data.to_csv("./artifacts/raw_filtered.csv", index=False)
    print("Saved filtered raw data to artifacts/raw_filtered.csv")
    return data

# Runs the function if script is executed directly
if __name__ == "__main__":
    data = prepare_data_and_artifacts(raw_data_path="/app/raw/raw_data.csv")

