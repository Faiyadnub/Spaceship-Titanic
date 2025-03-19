import numpy as np
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

from functions.helpers import split_and_insert, fill_missing_values_using_method

# Load the test data from Kaggle
df_test = pd.read_csv("data/test.csv")

# Convert the test data to DataFrame
data_test = pd.DataFrame(df_test)

# Preprocess data
data_test = data_test.astype(
    {col: "string" for col in data_test.select_dtypes(include=["object", "bool"]).columns})

# Split columns
data_test = split_and_insert(
    df=data_test,
    split_col="PassengerId",
    new_cols=["GroupId", "GroupSize"],
    spliter="_")

data_test = split_and_insert(
    df=data_test,
    split_col="Cabin",
    new_cols=["Deck", "CabinNumber", "Side"],
    spliter="/")

data_test = split_and_insert(
    df=data_test,
    split_col="Name",
    new_cols=["FirstName", "LastName"],
    spliter=" ")

# Drop columns
data_test = data_test.drop(
    columns=["PassengerId", "Cabin", "Name", "FirstName"])

# Map values
data_test["CryoSleep"] = data_test["CryoSleep"].map({"True": 1, "False": 0})
data_test["VIP"] = data_test["VIP"].map({"True": 1, "False": 0})

# cahnge data types
data_test["CabinNumber"] = data_test["CabinNumber"].astype(float)
data_test["Age"] = data_test["Age"].round().astype(float)

# Total spending
features_spending = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Calculate spending feature
data_test["TotalSpending"] = data_test[features_spending].sum(axis=1)
data_test["SpentMoney"] = (data_test["TotalSpending"] > 0).astype(int)
data_test["GroupSpentMoney"] = data_test.groupby("GroupId")["SpentMoney"].transform('max')

# Cap the extreme values to a specified percentile.
cap_test = data_test["TotalSpending"].quantile(0.95)
data_test["TotalSpending"] = np.where(
    data_test["TotalSpending"] > cap_test, cap_test, data_test["TotalSpending"])

# Group by side and deck and cabin number
spending_per_deck_test = data_test[data_test["CryoSleep"] != 1].groupby(
    ["Side", "Deck"])["TotalSpending"].mean().unstack()

# New column for traveling alone
data_test["TravelingAlone"] = data_test["GroupSize"] == "01"
data_test["TravelingAlone"] = data_test["TravelingAlone"].astype(int)

# Data mask
row_avg = data_test[features_spending].mean(axis=1)
child_under_13_mask = data_test["Age"] < 13

# Handling missing values
for col in features_spending:
    data_test[col] = data_test[col].where(~child_under_13_mask, 0)
    data_test[col] = data_test[col].fillna(row_avg)

# Data mask
missing_cryo_sleep = data_test["CryoSleep"].isna()
spend_zero = data_test["SpentMoney"] == 0
spend_not_zero = data_test["SpentMoney"] == 1
group_size_one = data_test["TravelingAlone"] == 1

# Handling missing values
data_test.loc[missing_cryo_sleep & spend_zero & group_size_one, "CryoSleep"] = 1
data_test.loc[missing_cryo_sleep & spend_not_zero, "CryoSleep"] = 0

# Data mask
missing_cryo_sleep = data_test["CryoSleep"].isna()
group_spend_money = data_test["GroupSpentMoney"] == 0
group_size_not_one = data_test["GroupSize"] != "01"

# Handling missing values
data_test.loc[missing_cryo_sleep & group_spend_money & group_size_not_one, "CryoSleep"] = 1
data_test = fill_missing_values_using_method(
    data_test, "GroupId", "CryoSleep", method="mode")

data_test = fill_missing_values_using_method(
    data_test, "GroupId", "HomePlanet", method="mode")

# Data mask
mask_europe = pd.isna(data_test["HomePlanet"]) & data_test["Deck"].isin(["A", "B", "C"])
mask_earth = pd.isna(data_test["HomePlanet"]) & (data_test["Deck"] == "G")

# Handling missing values
data_test.loc[mask_europe, "HomePlanet"] = "Europa"
data_test.loc[mask_earth, "HomePlanet"] = "Earth"

data_test = fill_missing_values_using_method(
    data_test, ["LastName"], "HomePlanet", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["Deck", "Side"], "HomePlanet", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["Deck", "CabinNumber", "Side"], "Destination", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["Deck", "Side"], "Destination", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["GroupId", "LastName"], "Destination", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["GroupId"], "Destination", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["LastName"], "Destination", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["GroupId"], "Side", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["LastName"], "Side", method="mode")
data_test = fill_missing_values_using_method(
    data_test, ["HomePlanet", "Destination"], "Side", method="mode")


# Define the decks available for each HomePlanet
decks_by_hp = {
    "Earth": ["E", "F", "G"],
    "Mars": ["D", "E", "F"],
    "Europa": ["A", "B", "C", "D", "E"]
}

# Define dict spending's for each side
spend_p_deck = spending_per_deck_test.T["P"].to_dict()
spend_s_deck = spending_per_deck_test.T["S"].to_dict()


# As we use function only once we will leave it here.
def fill_deck(row: pd.Series):
    """
    Fills the Deck value in the dataframe based on row of the home planet, total spending, and side.

    Parameters:
    - row (pd.Series): A row from the DataFrame containing HomePlanet, TotalSpending, Side, and Deck.

    Returns:
    - Union[str, float]: The closest deck based on the side and total spending, or the existing Deck value.
                         Returns np.nan if Side is neither S nor P.
    """
    h_planet = row["HomePlanet"]
    total_spending = row["TotalSpending"]
    side = row["Side"]

    possible_decks = decks_by_hp.get(h_planet, [])

    if pd.isna(row["Deck"]):
        # Get the available decks for the home planet
        if side == "S":
            spend_mean = spend_s_deck
        elif side == "P":
            spend_mean = spend_p_deck
        else:
            return np.nan

            # Find the closest deck based on total spending
        closest_deck = min(possible_decks, key=lambda deck: abs(spend_mean[deck] - total_spending))

        return closest_deck
    else:
        return row["Deck"]


# Fill missing values of deck feature
data_test["Deck"] = data_test.apply(fill_deck, axis=1)

# Change datatype
data_test["GroupId"] = data_test["GroupId"].astype(int)

# Linear regression for cabin number
decks = ["A", "B", "C", "D", "E", "F", "G"]
for deck in decks:
    deck_full_data = data_test[
        (data_test["Deck"] == deck) & (data_test["CabinNumber"].notnull())]
    x_full = deck_full_data["GroupId"]
    y_full = deck_full_data["CabinNumber"]
    x_full = x_full.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_full, y_full)
    deck_data_with_nan = data_test[
        (data_test["Deck"] == deck) & (data_test["CabinNumber"].isnull())]
    x_with_nan = deck_data_with_nan["GroupId"]
    x_with_nan = x_with_nan.values.reshape(-1, 1)

    if len(x_with_nan) > 0:
        predictions = model.predict(x_with_nan)
        rounded_predictions = np.round(predictions).astype(int)
        rounded_predictions = np.maximum(rounded_predictions, 1)

        data_test.loc[
            (data_test["Deck"] == deck) &
            (data_test["CabinNumber"].isnull()),
            "CabinNumber"
        ] = rounded_predictions

# Define bins for age grouping and grouping
bins = [0, 13, 18, 25, 30, 40, 50, float("inf")]
labels = ["0 - 13", "14 - 18", "19 - 25", "26 - 30", "31 - 40", "41 - 50", "51+"]

# Fill missing values using median
data_test = fill_missing_values_using_method(
    data_test,
    ["HomePlanet", "CryoSleep", "SpentMoney", "TravelingAlone"],
    "Age",
    method="median")

# Create age groups
data_test["AgeGroup"] = pd.cut(
    data_test["Age"],
    bins=bins,
    labels=labels,
    right=False)

# Drop columns
data_test = data_test.drop(columns=["VIP"])
data_test = data_test.drop(columns=["LastName"])

# Convert DataFrame to a dictionary
data_test = data_test.to_dict(orient="records")

# Send test data to the Flask API
api_response = requests.post("http://localhost:5000/predict", json=data_test)

# Print status code
print("Status Code:", api_response.status_code)

# Check if the request was successful
if api_response.status_code == 200:
    # Get the response data
    response_data = api_response.json()

    # Check if response_data is a list of dictionaries
    if isinstance(response_data, list) and all(isinstance(item, dict) for item in response_data):

        # Convert response data to a DataFrame
        df = pd.DataFrame(response_data)

        # Save DataFrame to a CSV file
        df.to_csv("data/api_predictions.csv", index=False)
        print("Predictions saved to 'predictions.csv'")

    else:
        print("Response data is not in the expected format.")

else:
    print("Failed to get predictions")
