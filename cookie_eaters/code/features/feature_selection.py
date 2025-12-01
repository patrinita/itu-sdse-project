
#removing irrelevant columns from the dataframe
data = data.drop(
    [
        "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen", "domain", "country", "visited_learn_more_before_booking", "visited_faq"
    ],
    axis=1
)