import pandas as pd
import pickle
from flask import Flask, request, render_template
import os
import numpy as np

# Create flask app
flask_app = Flask(__name__)

# Define the paths to the model, scaler, and imputer
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
imputer_path = os.path.join(os.path.dirname(__file__), 'imputer.pkl')

# Load the pickle model, scaler, and imputer
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
imputer = pickle.load(open(imputer_path, "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Check if the request has the file part
    if 'file' not in request.files:
        return render_template("index.html", prediction_text="No file part")

    file = request.files['file']

    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return render_template("index.html", prediction_text="No selected file")

    if file:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(file)

        # Preprocess the input data
        if 'type' in data.columns:
            data['type'] = data['type'].map({'white': 1, 'red': 0})

        # Handle missing values by imputing with the mean
        data = imputer.transform(data)

        # Scale the features
        features = scaler.transform(data)

        # Making prediction using the loaded model
        prediction = model.predict(features)

        # Clip the predictions to be between 1 and 10
        output = np.clip(prediction, 1, 10)

        # Round the output for display
        output = [round(pred, 2) for pred in output]

        return render_template("index.html", prediction_text=f"Predicted wine qualities are {output}")


if __name__ == "__main__":
    flask_app.run(debug=True)
