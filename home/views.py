from django.shortcuts import render
import numpy as np
import joblib

# Load the saved models
category_model = joblib.load("plant_category_model.pkl")
type_model = joblib.load("plant_type_model.pkl")
scaler = joblib.load("scaler.pkl")
label_enc_recommendation = joblib.load("label_encoder.pkl")
inverse_type_mapping = joblib.load("inverse_type_mapping.pkl")

def home(request):
    return render(request, 'index.html') 

def plant_recommender(request):
    recommended_category = None
    top_plant_types = []

    if request.method == "POST":
        # Get input values from form
        nitrogen = float(request.POST.get("nitrogen"))
        phosphorus = float(request.POST.get("phosphorus"))
        potassium = float(request.POST.get("potassium"))
        temperature = float(request.POST.get("temperature"))
        humidity = float(request.POST.get("humidity"))
        ph_level = float(request.POST.get("ph_level"))

        # Scale input data
        input_data = scaler.transform([[nitrogen, phosphorus, potassium, temperature, humidity, ph_level]])

        # Predict plant category (Leguminous/General)
        category_index = category_model.predict(input_data)[0]
        recommended_category = label_enc_recommendation.inverse_transform([category_index])[0]

        # Predict probabilities for plant types
        probabilities = type_model.predict_proba(input_data)[0]

        # Get indices of the top 3 plant types with the highest probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]  # Sort in descending order

        # Map indices to plant type names
        top_plant_types = [inverse_type_mapping.get(idx, "Unknown") for idx in top_indices]

    return render(request, "plant_recommender.html", {
        "recommended_category": recommended_category, 
        "plant_types": top_plant_types
    })
