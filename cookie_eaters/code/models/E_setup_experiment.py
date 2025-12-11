import datetime
import os
import shutil
import mlflow
import pandas as pd

def setup_mlflow(data, artifacts_path="./artifacts", mlruns_path="./mlruns"):
    
    # Checks directories exists
    os.makedirs(artifacts_path, exist_ok=True)
    os.makedirs(mlruns_path, exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    
    # Defines constants:
    experiment_name = datetime.datetime.now().strftime("%Y_%B_%d")
    data_gold_path = "./artifacts/train_data_gold.csv"
    data_version = "00000"
    artifact_path = "model"
    model_name = "lead_model"

    data.to_csv('./artifacts/train_data_gold.csv', index=False)

    mlflow.set_experiment(experiment_name)

    return data_gold_path, experiment_name

if __name__ == "__main__":
    # Load data 
    train_data = pd.read_csv("./artifacts/training_data.csv")  
    # Run MLflow setup
    setup_mlflow(train_data)
    print("MLflow setup completed!")