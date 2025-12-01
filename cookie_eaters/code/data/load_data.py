!dvc pull

import pandas as pd

print("Loading training data")

data = pd.read_csv("../../data/raw/raw_data.csv")

print("Total rows:", data.count())
display(data.head(5))

import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x: "%.3f" % x)


import json

data_columns = list(data.columns)
with open('./artifacts/columns_drift.json','w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv('./artifacts/training_data.csv', index=False)