# Importing main Libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Creating Flask app
app= Flask(__name__)


# Loading model and preprocessor

model = pickle.load(open('svc_model.pkl', 'rb'))
scaler= pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Getting the form data from the request object
    months_since_last_donation = int(request.form.get('months_since_last_donation'))
    months_since_first_donation = int(request.form.get('months_since_first_donation'))
    number_of_donations = int(request.form.get('number_of_donations'))

    features= np.array([months_since_last_donation, months_since_first_donation, number_of_donations])

    # Preprocessing
    feature_list= list(features)
    feature_sqrt= np.array([np.sqrt(x) for x in feature_list]).reshape(1,-1)
    feature_scaled=scaler.transform(feature_sqrt)
    feature_transformed= np.array(feature_scaled)

    # prediction
    prediction= model.predict_proba(feature_transformed)[:, 1] 

    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
