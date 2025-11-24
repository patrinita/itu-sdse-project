import pandas as pd

data = pd.read_csv("../../data/raw/raw_data.csv")

print("Total rows:", data.count())
display(data.head(5))
