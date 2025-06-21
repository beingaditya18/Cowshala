# 🐄 COWशाला – Cattle Breed & Disease Classifier with Gemini AI

**COWशाला** is an AI-powered Flask web application designed to assist farmers and researchers by identifying **cattle breeds** and **predicting diseases** from images. It combines a PyTorch-based deep learning model and **Gemini AI** to enrich the predictions with informative insights.

---

## 🚀 Features

### 🔬 Cattle Breed Classification
- Upload or link a cow image to identify its breed.
- Top-N breed predictions with probabilities.
- Gemini AI provides **key characteristics** and **common regions** for the predicted breed.

### 🦠 Disease Prediction
- Upload images of infected body parts (like skin, eyes, etc.).
- Returns possible diseases with confidence scores.
- Model trained using real symptom images (non-textual).

### 🤖 Gemini AI Integration
- Retrieves smart, summarized insights about cattle breeds using Google’s Gemini API.

### 🌐 Web + API Support
- Clean and responsive web UI.
- API endpoints for automation, integrations, or mobile apps.

---

## 📁 Project Structure
├── app.py # Main Flask app (breed + disease)
├── config.yaml # API keys & configs
├── models/
│ ├── cattle_breed_classifier_full_model.pth
│ ├── disease_prediction_model.pkl
│ └── classes.txt
├── templates/
│ ├── dashboard.html
│ ├── prediction.html
│ └── disease_result.html
├── static/
│ └── style.css

## ⚙️ Setup Instructions

### 1. 🔧 Install Requirements

```bash
pip install -r requirements.txt
2. 🧠 Download Models
Models will be downloaded automatically if not present:

cattle_breed_classifier_full_model.pth

classes.txt

disease_prediction_model.pkl

Make sure the models/ directory exists.

3. 🔑 Gemini API Key
Create a config.yaml file with your API key:

yaml
Copy
Edit
gemini_api_key: "YOUR_GEMINI_API_KEY"
▶️ Run the App
bash
Copy
Edit
python app.py
Open your browser at http://localhost:5001/



