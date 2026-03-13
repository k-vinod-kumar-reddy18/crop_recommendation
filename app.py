import streamlit as st
import pickle
import numpy as np
import requests
import os

# -------------------------------
# Load Model Safely
# -------------------------------
base_path = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_path, "crop_model.pkl"), "rb"))
le = pickle.load(open(os.path.join(base_path, "label_encoder.pkl"), "rb"))

# -------------------------------
# Weather API Key
# -------------------------------
API_KEY = "e96a5a5dd58dda99a62ffdf49ab98611"

# -------------------------------
# Weather Function (Safe)
# -------------------------------
def get_weather(city):

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            st.error("Weather data not found. Check city name or API key.")
            return None, None, None

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        rainfall = 0
        if "rain" in data:
            rainfall = data["rain"].get("1h", 0)

        return temperature, humidity, rainfall

    except:
        st.error("Unable to fetch weather data.")
        return None, None, None


# -------------------------------
# Multi Language Support
# -------------------------------
translations = {

"English":{
"title":"🌱 Smart Crop Recommendation System",
"city":"Enter City Name",
"N":"Nitrogen (N)",
"P":"Phosphorus (P)",
"K":"Potassium (K)",
"ph":"Soil pH",
"button":"Predict Crop",
"result":"Recommended Crop"
},

"Telugu":{
"title":"🌱 పంట సిఫార్సు వ్యవస్థ",
"city":"నగరం పేరు నమోదు చేయండి",
"N":"నైట్రోజన్",
"P":"ఫాస్ఫరస్",
"K":"పొటాషియం",
"ph":"మట్టి pH",
"button":"పంట సూచించు",
"result":"సిఫార్సు చేసిన పంట"
},

"Hindi":{
"title":"🌱 फसल सिफारिश प्रणाली",
"city":"शहर का नाम दर्ज करें",
"N":"नाइट्रोजन",
"P":"फॉस्फोरस",
"K":"पोटैशियम",
"ph":"मिट्टी pH",
"button":"फसल बताएं",
"result":"सिफारिश की गई फसल"
},

"Kannada":{
"title":"🌱 ಬೆಳೆ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ",
"city":"ನಗರದ ಹೆಸರು ನಮೂದಿಸಿ",
"N":"ನೈಟ್ರೋಜನ್",
"P":"ಫಾಸ್ಫರಸ್",
"K":"ಪೊಟ್ಯಾಸಿಯಂ",
"ph":"ಮಣ್ಣಿನ pH",
"button":"ಬೆಳೆ ಸೂಚಿಸಿ",
"result":"ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ"
},

"Tamil":{
"title":"🌱 பயிர் பரிந்துரை அமைப்பு",
"city":"நகரத்தின் பெயரை உள்ளிடவும்",
"N":"நைட்ரஜன்",
"P":"பாஸ்பரஸ்",
"K":"பொட்டாசியம்",
"ph":"மண் pH",
"button":"பயிர் கணிக்க",
"result":"பரிந்துரைக்கப்பட்ட பயிர்"
},

"Malayalam":{
"title":"🌱 വിള ശുപാർശ സംവിധാനം",
"city":"നഗരത്തിന്റെ പേര് നൽകുക",
"N":"നൈട്രജൻ",
"P":"ഫോസ്ഫറസ്",
"K":"പൊട്ടാസ്യം",
"ph":"മണ്ണിന്റെ pH",
"button":"വിള പ്രവചിക്കുക",
"result":"ശുപാർശ ചെയ്ത വിള"
}

}

# -------------------------------
# Language Selection
# -------------------------------
language = st.sidebar.selectbox(
"🌍 Select Language",
["English","Telugu","Hindi","Kannada","Tamil","Malayalam"]
)

t = translations[language]

# -------------------------------
# UI
# -------------------------------
st.title(t["title"])

city = st.text_input(t["city"])

st.subheader("Soil Parameters")

N = st.number_input(t["N"], min_value=0.0)
P = st.number_input(t["P"], min_value=0.0)
K = st.number_input(t["K"], min_value=0.0)
ph = st.number_input(t["ph"], min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button(t["button"]):

    if city == "":
        st.warning("Please enter city name")

    else:

        temperature, humidity, rainfall = get_weather(city)

        if temperature is None:
            st.stop()

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction = model.predict(features)

        crop = le.inverse_transform(prediction)

        st.success(f"{t['result']}: {crop[0]}")

        st.write("🌡 Temperature:", temperature)
        st.write("💧 Humidity:", humidity)
        st.write("🌧 Rainfall:", rainfall)
