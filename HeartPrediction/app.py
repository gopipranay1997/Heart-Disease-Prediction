# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Load the CatBoost CLassifier model
filename = 'finalized_model.pickle'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        LIMIT_BAL = int(request.form['LIMIT_BAL'])
        BILL_AMT1 = int(request.form['BILL_AMT1'])
        BILL_AMT2 = int(request.form['BILL_AMT2'])
        BILL_AMT3 = int(request.form['BILL_AMT3'])
        BILL_AMT4 = int(request.form['BILL_AMT4'])
        BILL_AMT5 = int(request.form['BILL_AMT5'])
        BILL_AMT6 = int(request.form['BILL_AMT6'])
        PAY_AMT1 = int(request.form['PAY_AMT1'])
        PAY_AMT2 = int(request.form['PAY_AMT2'])
        PAY_AMT3 = int(request.form['PAY_AMT3'])
        PAY_AMT4 = int(request.form['PAY_AMT4'])
        PAY_AMT5 = int(request.form['PAY_AMT5'])
        PAY_AMT6 = int(request.form['PAY_AMT6'])
        SEX = int(request.form['SEX'])
        EDUCATION = request.form.get('EDUCATION')
        MARRIAGE = request.form.get('MARRIAGE')
        AGE = request.form.get('AGE')
        PAY_0 = request.form.get('PAY_0')
        PAY_2 = request.form.get('PAY_2')
        PAY_3 = request.form.get('PAY_3')
        PAY_4 = request.form.get('PAY_4')
        PAY_5 = request.form.get('PAY_5')
        PAY_6 = request.form.get('PAY_6')
        
        data = np.array([[LIMIT_BAL,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
    
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
