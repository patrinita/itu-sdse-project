import mlflow
import mlflow.pyfunc
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score,confusion_matrix, classification_report, accuracy_score
from E_setup_experiment import setup_mlflow

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def load_and_split_data():
    data = pd.read_csv("./artifacts/train_data_gold.csv")

    y = data["lead_indicator"]
    X = data.drop(columns=["lead_indicator"])

    return train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

def train_lr_with_mlflow(X_train, y_train, X_test, y_test, experiment_name):
    
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"
        
        params = {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "penalty": ["none", "l1", "l2", "elasticnet"],
            "C": [100, 10, 1.0, 0.1, 0.01],
        }

        model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")

        joblib.dump(value=model, filename=lr_model_path)

        mlflow.pyfunc.log_model("model", python_model=lr_wrapper(best_model))

    return {
        "model_grid": model_grid,
        "model": model,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "model_path": lr_model_path,
    }


def evaluate(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

def main():
    _, experiment_name = setup_mlflow()

    X_train, X_test, y_train, y_test = load_and_split_data()

    results = train_lr_with_mlflow(
        X_train, y_train, X_test, y_test, experiment_name
    )

    train_metrics = evaluate(y_train, results["y_pred_train"])
    test_metrics = evaluate(y_test, results["y_pred_test"])

    print("Best Logistic Regression params:")
    print(results["model_grid"].best_params_)

    print("\n Accuracy train:", train_metrics["accuracy"])
    print("Accuracy test:", test_metrics["accuracy"])

    print("\nTest classification report:")
    print(pd.DataFrame(test_metrics["classification_report"]).T)
    
    return {
        results["model_path"]: test_metrics["classification_report"]
    }


if __name__ == "__main__":
    main()
