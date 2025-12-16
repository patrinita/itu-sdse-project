import json
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import mlflow
import pandas as pd
from code.models.E_setup_experiment import setup_mlflow

artifact_path = "model"
model_name = "lead_model"


#get experiment model results
def get_best_experiment(experiment_name):
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    return experiment_best

#metrics
def get_best_model(path="./artifacts/model_results.json"):#added
    with open(path, "r") as f:
        model_results = json.load(f)
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T
    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    
    return best_model


def main(experiment_name, model_results_path="./artifacts/model_results.json"):

    experiment_best = get_best_experiment(experiment_name)
    print(f"Best experiment: {experiment_best}")
    
    best_model = get_best_model(model_results_path)
    print(f"Best model: {best_model}")

    return {
        "experiment_best": experiment_best,
        "best_model": best_model
    }


if __name__ == "__main__":
    _, experiment_name = setup_mlflow()
    
    outputs = main(experiment_name)
    