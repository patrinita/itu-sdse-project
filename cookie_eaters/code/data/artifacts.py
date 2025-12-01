import os
import shutil
from pprint import pprint

# shutil.rmtree("./artifacts",ignore_errors=True)
os.makedirs("artifacts",exist_ok=True)
print("Created artifacts directory")

import json

data_columns = list(data.columns)
with open('./artifacts/columns_drift.json','w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv('./artifacts/training_data.csv', index=False)

data.to_csv('./artifacts/train_data_gold.csv', index=False)