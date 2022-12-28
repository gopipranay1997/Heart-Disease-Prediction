# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'classifier.pickle'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        thalach = int(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        sex = request.form.get('sex')
        cp1 = request.form.get('cp1')
        cp2 = request.form.get('cp2')
        cp3 = request.form.get('cp3')
        cp4 = request.form.get('cp4')
        cp5 = request.form.get('cp5')
        cp6 = request.form.get('cp6')
        cp7 = request.form.get('cp7')
        cp8 = request.form.get('cp8')
        
        data = np.array([[age,trestbps,chol,fbs,thalach,oldpeak,sex,cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=False)


