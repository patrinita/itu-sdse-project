from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint
import pandas as pd
from F_load_train_data import prepare_training_data


def build_model_grid(): #added this
    model = XGBRFClassifier(random_state=42)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

    model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)

    return model_grid #added this and the code below

def main(data):
    
    data = pd.read_csv("./artifacts/train_data_gold.csv")
    
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )
    y_train

    model_grid = build_model_grid()
    model_grid.fit(X_train, y_train) #didnt add this

if __name__ == "__main__":
    main()