import pandas as pd
from E_setup_experiment import data_gold_path
from code.data.A_helper_functions import create_dummy_cols

#should also be in a function:
def prepare_training_data(data): 

    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)


    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
        print(f"Changed column {col} to float")
    
    return data

def main():
    # Load dataset
    data = pd.read_csv(data_gold_path)
    print(f"Training data length: {len(data)}")
    print(data.head(5))

    # Process data
    processed = prepare_training_data(data)

    # writes the gold dataset
    processed.to_csv("./artifacts/train_data_gold.csv", index=False)
    print("Saved processed training data to ./artifacts/train_data_gold.csv")

if __name__ == "__main__":
    main()
