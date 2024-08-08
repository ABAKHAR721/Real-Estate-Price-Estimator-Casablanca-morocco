from flask import Flask, request, render_template
import pickle
import pandas as pd


app = Flask(__name__, template_folder='Flask/templates')
# Load the model and preprocessor
with open('./Flask/model/casablanca_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./Flask/model/casablanca_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)


# 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', price_estimate=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data
    age = request.form.get('Age', 0)
    current_state=request.form.get('Current_state', "")
    area=float(request.form['Area'])
    data = {
        'Type': request.form['Type'],
        'Localisation': request.form['Localisation'],
        'Area': float(request.form['Area']),
        'Rooms': int(request.form['Rooms']),
        'Bedrooms': int(request.form['Bedrooms']),
        'Bathrooms': int(request.form['Bathrooms']),
        'Floor': int(request.form['Floor']),
        'Current_state': current_state,  # Capture this field
        'Age': age   # Capture this field
    }

    df = pd.DataFrame([data])

    # Ensure all required columns are in the DataFrame
    expected_columns = preprocessor.get_feature_names_out()
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with 0

    df_preprocessed = preprocessor.transform(df)
    prediction = model.predict(df_preprocessed)


    return render_template('index.html',
                           price_estimate=float(prediction[0]),
                           price_estimate_per_m2=float(prediction[0] // area) )

if __name__ == '__main__':
    app.run(debug=True)
