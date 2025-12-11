import json #added
from pprint import pprint #added

#save the model result
def save_column_list(X_train, path):#added
    with open(path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        pprint(columns)
        json.dump(columns, columns_file)
    print('Saved column list to', path)

def save_model_results(model_results, path):
    with open(path, 'w+') as results_file:
        json.dump(model_results, results_file)
    print('Saved model results to', path)

def main():
    column_list_path = './artifacts/columns_list.json'
    model_results_path = './artifacts/model_results.json'

    save_column_list(X_train, column_list_path)
    save_model_results(model_results, model_results_path)


if __name__ == "__main__":
    main()