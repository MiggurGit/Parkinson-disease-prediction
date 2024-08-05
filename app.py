from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('parkinsons_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect data from form
        form_data = request.form
        features = [
            float(form_data['MDVP:Fo(Hz)']),
            float(form_data['MDVP:Fhi(Hz)']),
            float(form_data['MDVP:Flo(Hz)']),
            float(form_data['MDVP:Jitter(%)']),
            float(form_data['MDVP:Jitter(Abs)']),
            float(form_data['MDVP:RAP']),
            float(form_data['MDVP:PPQ']),
            float(form_data['Jitter:DDP']),
            float(form_data['MDVP:Shimmer']),
            float(form_data['MDVP:Shimmer(dB)']),
            float(form_data['Shimmer:APQ3']),
            float(form_data['Shimmer:APQ5']),
            float(form_data['MDVP:APQ']),
            float(form_data['Shimmer:DDA']),
            float(form_data['NHR']),
            float(form_data['HNR']),
            float(form_data['RPDE']),
            float(form_data['D2']),
            float(form_data['DFA']),
            float(form_data['spread1']),
            float(form_data['spread2']),
            float(form_data['PPE'])
        ]
        
        prediction = model.predict([features])[0]
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
