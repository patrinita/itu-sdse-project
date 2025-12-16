import json
from pprint import pprint
from code.models.H_sklearn_train_and_evaluate import main as train_main
import pandas as pd

def save_artifacts(X_train, model_results, column_list_path="./artifacts/columns_list.json", model_results_path="./artifacts/model_results.json"):
   
    """
    Saves training column names and model results to disk.

    Returns:
        dict with paths to saved artifacts
    """
    
    # Save column list
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        pprint(columns)
        json.dump(columns, columns_file)

    print("Saved column list to", column_list_path)

    # Save model results
    with open(model_results_path, "w+") as results_file:
        json.dump(model_results, results_file)

    print("Saved model results to", model_results_path)

    return {
        "columns_path": column_list_path,
        "model_results_path": model_results_path,
        "columns": columns,
        "model_results": model_results,
    }

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("./artifacts/train_data_gold.csv")
    
    # Run training/evaluation to get outputs
    outputs = train_main(data)
    
    # Save artifacts
    artifacts = save_artifacts(outputs["X_train"], outputs["model_results"])
    print("Artifacts saved:", artifacts)