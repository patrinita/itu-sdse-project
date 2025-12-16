import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
from code.data.A_helper_functions import describe_numeric_col, impute_missing_values

def clean_and_preprocess_data(data, artifacts_path="./artifacts"):
    """
    Cleans, imputes, standardizes, and prepares training data.
    """
    #replacing values 
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]

    #added to check if source columnn is empty
    if data.empty:
        raise ValueError("Data is empty after filtering source == 'signup'")

    result=data.lead_indicator.value_counts(normalize = True)

    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)

    #fix data types
    vars = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")


    # Continuous annd categorical variables missing values
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    print("\nContinuous columns: \n")
    pprint(list(cont_vars.columns), indent=4)
    print("\n Categorical columns: \n")
    pprint(list(cat_vars.columns), indent=4)


    #handling outliers
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv('./artifacts/outlier_summary.csv')

    #impute data
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")


    # Continuous and categorical variables missing values
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T

    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T

    #data standardization
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    scaler_path = "./artifacts/scaler.pkl"
    joblib.dump(value=scaler, filename=scaler_path)
    print("Saved scaler in artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    

    #combine the data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    print(f"Data cleansed and combined.\nRows: {len(data)}")
    

    data_columns = list(data.columns)
    with open('./artifacts/columns_drift.json','w+') as f:           
        json.dump(data_columns,f)
        
    data.to_csv('./artifacts/training_data.csv', index=False)
    print("Saved training_data.csv in artifacts")

    return data
