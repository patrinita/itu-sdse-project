import os
import shutil
from pprint import pprint

# shutil.rmtree("./artifacts",ignore_errors=True)
os.makedirs("artifacts",exist_ok=True)
print("Created artifacts directory")

import os
import shutil

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)