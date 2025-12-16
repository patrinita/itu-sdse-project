import pandas as pd
from code.data.C_preprocessing import clean_and_preprocess_data

def feature_engineering(data, artifacts_path="./artifacts"):
    """
    Performs feature engineering on the input DataFrame.
    """
    
    #removing irrelevant columns from the dataframe
    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen", "domain", "country", "visited_learn_more_before_booking", "visited_faq"
        ],
        axis=1
    )

    #Group source values into broader categories

    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
            'fb' : 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
            }

    data['bin_source'] = data['source'].map(mapping)

    return data

