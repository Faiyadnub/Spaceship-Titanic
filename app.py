from flask import Flask, request, jsonify
import joblib
import pandas as pd

# App name
app = Flask(__name__)

# Load the saved model
model = joblib.load("ml_models/best_model.pkl")


# Add a route for the home page
@app.route("/", methods=["GET"])
def home():
    return ("<h1>Spaceship Titanic Model API</h1>"
            "<p>Your file need to be placed in data folder and name test.csv</p>"
            "<p>Use <b>python predict.py</b> to get predictions.</p>"
            "<p>Predictions will be saved in data folder</p>")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive test data from the POST request
        data_json = request.get_json(force=True)
        data_test = pd.DataFrame(data_json)

        # Check if preprocessing fill all missing values
        if data_test.isnull().values.any():
            raise ValueError(f"Null values: {data_test.isnull().sum()}")

        # Check for required columns
        required_columns = [
            "GroupSize", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "TotalSpending", "HomePlanet", "CryoSleep", "CabinNumber", "Side", "Destination",
            "TravelingAlone", "SpentMoney", "GroupSpentMoney", "Deck", "AgeGroup"
]
        missing_columns = [col for col in required_columns if col not in data_test.columns]

        # Check if all columns are valid
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Make predictions
        predictions = model.predict(data_test)

        # Convert GroupId and GroupSize to strings and fill missing 0 in the front of string
        data_test["GroupId"] = data_test["GroupId"].astype(str).str.zfill(4)
        data_test["GroupSize"] = data_test["GroupSize"].astype(str).str.zfill(2)

        # Join back PassengerId column
        data_test["PassengerId"] = data_test["GroupId"] + '_' + data_test["GroupSize"]

        # Prepare submission DataFrame
        submission_df = pd.DataFrame({
            "PassengerId": data_test["PassengerId"],
            "Transported": (predictions > 0.5).astype(bool)
        })

        # Return predictions in JSON format
        return jsonify(submission_df.to_dict(orient="records"))
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
