import mlflow
from code.data.A_helper_functions import wait_until_ready
from code.models.K_check_production_model import get_production_model, evaluate_production_model
from code.models.J_mlflow_model_selection import main as select_model_main
from code.models.E_setup_experiment import setup_mlflow

def compare_and_register_model(experiment_best, model_name="lead_model", artifact_path="model"):
    # Get production info
    prod_model = get_production_model(model_name)
    prod_model_exists, prod_model_run_id, prod_model_version = evaluate_production_model(
        prod_model, model_name
    )

    train_model_score = experiment_best["metrics.f1_score"]
    model_details = {}
    model_status = {}
    run_id = None

    # Compare with prod model
    if prod_model_exists:
        data = mlflow.get_run(prod_model_run_id)
        prod_model_score =data.data.metrics["f1_score"]

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        if train_model_score > prod_model_score:
            print("Registering new model")
            run_id = experiment_best["run_id"]
    else:
        print("No model in production")
        run_id = experiment_best["run_id"]

    # Register model if needed
    if run_id is not None:
        print(f'Best model found: {run_id}')
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)

    return model_details, prod_model

if __name__ == "__main__":

    # Get best experiment output from previous step
    _, experiment_name = setup_mlflow()
    selection_outputs = select_model_main(experiment_name)
    experiment_best = selection_outputs["experiment_best"]

    # Compare and register
    model_details, prod_model = compare_and_register_model(experiment_best)

    print(model_details)

   
