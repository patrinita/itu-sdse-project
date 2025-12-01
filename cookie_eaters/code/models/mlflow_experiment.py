#get experiment model results
experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
experiment_ids

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1
).iloc[0]
experiment_best

#metrics

import json

with open("./artifacts/model_results.json", "r") as f:
    model_results = json.load(f)
results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T
results_df

best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
print(f"Best model: {best_model}")