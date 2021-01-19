# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from csv import writer

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')


# Bind predict function to URL
@app.route('/predict', methods=['POST'])
def predict():
    # Put all form entries values in a list
    features = [i for i in request.form.values()]

    # Convert features to array
    array_features = [np.array(features)]

    # Predict features
    row = pd.DataFrame(array_features)

    # Predict output = probability of 1
    prediction = model.predict_proba(row)

    # Predict output = [0 1]
    pred = model.predict(row)

    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output) * 100) + '%'

    features.append(pred[0])

    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open('event.csv', 'a') as fd:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(fd)

        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(features)

    # Check the output values and retrieve the result with html tag based on the value
    if pred == 1:
        return render_template('Heart Disease Classifier.html',
                               result=f'You have chance of having Heart Disease: {pred}.\nProbability of having Heart Disease is {output}')
    else:
        return render_template('Heart Disease Classifier.html',
                               result=f'You are safe: {pred}.\n Probability of having Heart Disease is {output}')


if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
