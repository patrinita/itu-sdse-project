import datetime
import os
import shutil
import mlflow
import pandas as pd

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date
artifact_path = "model"
model_name = "lead_model"

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

data.to_csv('./artifacts/train_data_gold.csv', index=False)


mlflow.set_experiment(experiment_name)