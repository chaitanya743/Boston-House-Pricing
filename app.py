import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os  # Add this import for working with file paths

app = Flask(__name__)

# Construct the absolute file path for the pkl file
pkl_file_path = os.path.join(os.path.dirname(__file__), 'newregmodel.pkl')

# Load the new model and scaler
new_regmodel = pickle.load(open(pkl_file_path, 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get data from JSON input
        data = request.json['data']
        print(data)

        # Transform the input data using the scaler
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

        # Make prediction using the loaded model
        output = new_regmodel.predict(new_data)
        print(output[0])

        return jsonify(output[0])

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from HTML form
        data = [float(x) for x in request.form.values()]

        # Transform the input data using the scaler
        final_input = scaler.transform(np.array(data).reshape(1, -1))

        # Make prediction using the loaded model
        output = new_regmodel.predict(final_input)[0]

        return render_template("home.html", prediction_text="The House price prediction is {}".format(output))

    except Exception as e:
        return render_template("home.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
