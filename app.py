from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

# Lists of unique countries and crops the model was trained on
COUNTRIES = [
    'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 
    'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 
    'Burkina Faso', 'Burundi', 'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia', 
    'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 
    'Estonia', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 
    'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica', 
    'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Libya', 'Lithuania', 
    'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 
    'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 
    'Niger', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 'Qatar', 
    'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia', 'South Africa', 'Spain', 
    'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 
    'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe'
]

CROPS = [
    'Cassava', 'Maize', 'Plantains and others', 'Potatoes', 'Rice, paddy', 
    'Sorghum', 'Soybeans', 'Sweet potatoes', 'Wheat', 'Yams'
]

# Loading ML models
try:
    dtr = pickle.load(open('dtr.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
except Exception as e:
    print(f"Error loading pickle models: {e}")
    dtr = None
    preprocessor = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'index.html', 
        countries=COUNTRIES, 
        crops=CROPS,
        selected_year=2013
    )

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from form
            Year = request.form.get('Year', 2013)
            average_rain_fall_mm_per_year = request.form.get('average_rain_fall_mm_per_year', '')
            pesticides_tonnes = request.form.get('pesticides_tonnes', '')
            avg_temp = request.form.get('avg_temp', '')
            Area = request.form.get('Area', '')
            Item = request.form.get('Item', '')

            # Basic input validation
            if not average_rain_fall_mm_per_year or not pesticides_tonnes or not avg_temp:
                raise ValueError("All fields must be filled out.")

            # Make features array
            features = np.array(
                [[Year, float(average_rain_fall_mm_per_year), float(pesticides_tonnes), float(avg_temp), Area, Item]], 
                dtype=object
            )
            
            # Predict
            transformed_features = preprocessor.transform(features)
            raw_prediction = dtr.predict(transformed_features)[0]
            
            # Format prediction (e.g. 3,066 hg/ha)
            prediction = f"{raw_prediction:,.0f} hg/ha"
            error = None

        except ValueError as ve:
            prediction = None
            error = str(ve)
        except Exception as e:
            prediction = None
            error = f"An unexpected error occurred during prediction: {e}"

        return render_template(
            'index.html',
            countries=COUNTRIES,
            crops=CROPS,
            prediction=prediction,
            error=error,
            selected_year=int(Year) if Year else 2013,
            selected_rain=average_rain_fall_mm_per_year,
            selected_pesticides=pesticides_tonnes,
            selected_temp=avg_temp,
            selected_area=Area,
            selected_item=Item
        )

if __name__ == "__main__":
    app.run(debug=True)