from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
#@app.route('/test')
#def test():
    #return "Flask is being used for Development"


@app.route('/')
def home():
    return render_template('original.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            bmi = float(request.form['bmi'])
            Children_number = float(request.form['Children_number'])
            Smoker= float(request.form['Smoker'])
            Region = float(request.form['Region'])
            
            pred_args = [age,sex,bmi,Children_number,Smoker,Region]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            mul_reg = open('model.pkl','rb')
            ml_model = joblib.load(mul_reg)
            model_predcition = ml_model.predict(pred_args_arr)
            model_predcition = round(float(model_predcition),2)
        except valueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_predcition)

if __name__ == '__main__':
    app.run()
