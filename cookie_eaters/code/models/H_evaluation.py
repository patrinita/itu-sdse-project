from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd


#test accuracy
def best_params(model_grid): #added this one

    best_model_xgboost_params = model_grid.best_params_
    
    return best_model_xgboost_params

def get_predictions(model_grid, X_train, X_test): #added

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    return y_pred_train, y_pred_test #added

#look at this again because conf_matrix is double...
#performance overview
def performance(model_grid, X_test, y_test, X_train, y_train): #added
    
    y_pred_test = model_grid.predict(X_test)#added
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    
    y_pred_train = model_grid.predict(X_train)
    conf_matrix = confusion_matrix(y_train, y_pred_train)

    
    return y_pred_train, y_pred_test #added


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
    best_params(model_grid)
        # ###add the print statements
    # print("Best xgboost params")
    # pprint(best_model_xgboost_params)
    get_predictions(model_grid, X_train, X_test)
    
    # print("Accuracy train", accuracy_score(y_pred_train, y_train ))
    # print("Accuracy test", accuracy_score(y_pred_test, y_test))
    # print("Accuracy train", accuracy_score(y_pred_train, y_train))
    # print("Accuracy test", accuracy_score(y_pred_test, y_test))
    # ###
    performance(model_grid, X_test, y_test, X_train, y_train)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train),'\n')
    print(classification_report(y_test, y_pred_test),'\n')
    save_best_model_and_results(model_grid, y_train, y_pred_train)
    

if __name__ == "__main__":
    main()