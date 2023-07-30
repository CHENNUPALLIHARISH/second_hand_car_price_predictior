from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

df = pd.read_csv('dataset2.csv')  
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pipeline and the final model
pipeline = joblib.load('final_model.pkl')

@app.route('/')
def home():
  
    unique_brands = df['company'].unique()
    unique_models = df['name'].unique()
    unique_fuel_types = df['fuel_type'].unique()

    return render_template('index.html', companies=unique_brands, car_models=unique_models, fuel_types=unique_fuel_types, years=range(1990, 2023))

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['company']
    model_name = request.form['car_models']
    year = int(request.form['year'])
    kms = int(request.form['kms_no']) 
    fuel_type = request.form['fuel_type']

   
    input_data = pd.DataFrame({'company': [brand], 'name': [model_name], 'year': [year], 'kms_no': [kms], 'fuel_type': [fuel_type]})

 
    predicted_price = pipeline.predict(input_data)[0]

  
    return jsonify(predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
