from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('model.pkl')
cols = ['age', 'job', 'marital', 'education', 'housing', 'loan','contact', 'month', 'day_of_week',
        'default', 'campaign', 'pdays', 'previous', 'poutcome','emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    try:
        if int(prediction) == 0:
            prediction ='NO suscribe to term deposit'
        elif int(prediction) == 1:
            prediction = 'Suscribe to term deposit!!'
    except ValueError:
        prediction = 'Data Format Error'
    return render_template('result.html',pred='Expected conversion will be - {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

