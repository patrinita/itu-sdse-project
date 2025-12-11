import json
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)
        
#get experiment model results
def get_experiment_best(experiment_name):#added
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    return experiment_best#changed

#metrics
def load_model_results(path="./artifacts/model_results.json"):#added
    with open(path, "r") as f:
        model_results = json.load(f)
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T
    return model_results, results_df #changed

def get_best_model_name(results_df):#added
    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    return best_model#changed

def main():#added all lines below:
    experiment_best = get_experiment_best(experiment_name)

    model_results, results_df = load_model_results("./artifacts/model_results.json")

    best_model = get_best_model_name(results_df)
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()