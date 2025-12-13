import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import cohen_kappa_score, f1_score
import joblib

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def run_mlflow_experiment():#added
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"

        params = {
                'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                'penalty':  ["none", "l1", "l2", "elasticnet"],
                'C' : [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)


        # log artifacts
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")
        
        # store model for model interpretability
        joblib.dump(value=model, filename=lr_model_path)
            
        # Custom python model for predicting probability 
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))
    return model_grid, y_pred_train, y_pred_test, lr_model_path #added


def print_best_params(model_grid):#added
    best_model_lr_params = model_grid.best_params_
    print("Best lr params")
    pprint(best_model_lr_params)

def print_accuracies(y_train, y_test, y_pred_train, y_pred_test):#added
    print("Accuracy train:", accuracy_score(y_pred_train, y_train ))
    print("Accuracy test:", accuracy_score(y_pred_test, y_test))

def print_test_performance(y_test, y_pred_test):#added
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test),'\n')

def print_train_performance(y_train, y_pred_train):#added
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train),'\n')

def update_model_results(model_results, lr_model_path, y_test, y_pred_test):#added
    model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)#moved this line here
    model_results[lr_model_path] = model_classification_report
    print(model_classification_report["weighted avg"]["f1-score"])

def main():#added all lines below
    model_grid, y_pred_train, y_pred_test, lr_model_path = run_mlflow_experiment()

    print_best_params(model_grid)

    print_accuracies(y_train, y_test, y_pred_train, y_pred_test)

    print_test_performance(y_test, y_pred_test)
    print_train_performance(y_train, y_pred_train)

    update_model_results(model_results, lr_model_path, y_test, y_pred_test)


if __name__ == "__main__":
    main()