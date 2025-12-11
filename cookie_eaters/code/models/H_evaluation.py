from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json


#test accuracy
def print_best_params(model_grid): #added this one
    best_model_xgboost_params = model_grid.best_params_
    print("Best xgboost params")
    pprint(best_model_xgboost_params)

def get_predictions(model_grid, X_train, X_test): #added
    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)
    return y_pred_train, y_pred_test #added
def print_accuracy(y_train, y_test, y_pred_train, y_pred_test): #added
    print("Accuracy train", accuracy_score(y_pred_train, y_train ))
    print("Accuracy test", accuracy_score(y_pred_test, y_test))

#performance overview

def performance_test(model_grid, X_test, y_test): #added
    y_pred_test = model_grid.predict(X_test)#added

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test),'\n')
return y_pred_test #added

def performance_train(model_grid, X_train, y_train):
    y_pred_train = model_grid.predict(X_train)
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train),'\n')
    return y_pred_train #added

def print_accuracy(y_train, y_test, y_pred_train, y_pred_test):#added all 3 lines
    print("Accuracy train", accuracy_score(y_pred_train, y_train))
    print("Accuracy test", accuracy_score(y_pred_test, y_test))

#save the best model
def save_best_model_and_results(model_grid, y_train, y_pred_train): #added
    xgboost_model = model_grid.best_estimator_
    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    xgboost_model.save_model(xgboost_model_path)

    model_results = {
        xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
    }
    return model_results #added

def main():#added all lines below
    print_best_params(model_grid)

    y_pred_test = performance_test(model_grid, X_test, y_test)
    y_pred_train = performance_train(model_grid, X_train, y_train)

    print_accuracy(y_train, y_test, y_pred_train, y_pred_test)

    save_best_model_and_results(model_grid, y_train, y_pred_train)


if __name__ == "__main__":
    main()