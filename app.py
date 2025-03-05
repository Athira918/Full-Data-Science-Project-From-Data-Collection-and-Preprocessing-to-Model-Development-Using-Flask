
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])  # Convert input to DataFrame
    prediction = model.predict(df)
    
    return jsonify({'Predicted Price': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
