#test accuracy
from sklearn.metrics import accuracy_score

best_model_xgboost_params = model_grid.best_params_
print("Best xgboost params")
pprint(best_model_xgboost_params)

y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)
print("Accuracy train", accuracy_score(y_pred_train, y_train ))
print("Accuracy test", accuracy_score(y_pred_test, y_test))

#performance overview
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Test actual/predicted\n")
print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_test, y_pred_test),'\n')

conf_matrix = confusion_matrix(y_train, y_pred_train)
print("Train actual/predicted\n")
print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_train, y_pred_train),'\n')

#save the model result
column_list_path = './artifacts/columns_list.json'
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    pprint(columns)
    json.dump(columns, columns_file)

print('Saved column list to ', column_list_path)

model_results_path = "./artifacts/model_results.json"
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)