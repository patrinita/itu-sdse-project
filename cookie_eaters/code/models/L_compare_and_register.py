import mlflow
import dagger
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# Compare prod and best trained model
def compare_and_register_model():#added
    train_model_score = experiment_best["metrics.f1_score"]
    model_details = {}
    model_status = {}
    run_id = None

    if prod_model_exists:
        data, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = data[1]["metrics.f1_score"]

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        if train_model_score>prod_model_score:
            print("Registering new model")
            run_id = experiment_best["run_id"]
    else:
        print("No model in production")
        run_id = experiment_best["run_id"]

    print(f"Registered model: {run_id}")


# Register best model

    if run_id is not None:
        print(f'Best model found: {run_id}')

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id,
            artifact_path=artifact_path
        )
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)
        print(model_details)

    return model_details, model_status, run_id#added

def main():#added all lines below:
    compare_and_register_model()


if __name__ == "__main__":
    main()