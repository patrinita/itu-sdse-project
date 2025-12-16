import pandas as pd
from code.data.A_helper_functions import create_dummy_cols
from code.data.B_setup_data import prepare_data_and_artifacts
from code.data.C_preprocessing import clean_and_preprocess_data
from code.features.D_feature_engineering import feature_engineering

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
    raw_data = prepare_data_and_artifacts()
    cleaned_data = clean_and_preprocess_data(raw_data)
    engineered_data = feature_engineering(cleaned_data)

    # Process data
    processed = prepare_training_data(engineered_data)

    print(f"Training data length: {len(processed)}")
    print(processed.head(5))

    # writes the gold dataset
    processed.to_csv("./artifacts/train_data_gold.csv", index=False)
    print("Saved processed training data to ./artifacts/train_data_gold.csv")
    
    return processed

if __name__ == "__main__":
    main()
