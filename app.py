# app.py
from flask import Flask, request, jsonify, render_template, session
import pickle
import numpy as np
import pandas as pd  # <-- Needed to construct DataFrame with feature names
import os
from chatbot import ChatbotResponse
from symptom_categories import categorize_symptoms

app = Flask(__name__, static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')

# Load the trained model components
with open('disease_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
symptoms = model_data['symptoms']
disease_names = model_data['disease_names']
imputer = model_data['imputer']

# Categorize symptoms
categorized_symptoms = categorize_symptoms(symptoms)

# Initialize the chatbot
chatbot = ChatbotResponse()

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms, categorized_symptoms=categorized_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    # Get selected symptoms from the form
    selected_symptoms = request.form.getlist('symptoms')
    
    # Create feature vector (all 0s initially)
    feature_vector = np.zeros(len(symptoms))
    
    # Set selected symptoms to 1
    for symptom in selected_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            feature_vector[index] = 1

    # Apply the same imputation used during training
    imputed_vector = imputer.transform([feature_vector])[0]

    # Convert to DataFrame with valid feature names
    input_df = pd.DataFrame([imputed_vector], columns=symptoms)

    # Make prediction
    predicted_class = model.predict(input_df)[0]
    prediction = disease_names[predicted_class]

    # Get probabilities for all classes
    prediction_proba = model.predict_proba(input_df)[0]

    # Get top 3 diseases with their probabilities
    top_indices = prediction_proba.argsort()[-3:][::-1]
    top_diseases = [(disease_names[i], float(prediction_proba[i])) for i in top_indices]

    result = {
        'prediction': prediction,
        'top_diseases': top_diseases
    }
    
    return jsonify(result)

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user message from the request
    user_message = request.json.get('message', '')
    
    # Process the message with the chatbot
    response = chatbot.process_message(user_message)
    
    # If the chatbot is in the predicting state, extract the prediction results
    if chatbot.current_state == "predicting":
        # Extract the confirmed symptoms for display
        confirmed_symptoms = [s.replace('_', ' ').title() for s in chatbot.confirmed_symptoms]
        
        # Create a feature vector for the confirmed symptoms
        feature_vector = np.zeros(len(symptoms))
        for symptom in chatbot.confirmed_symptoms:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                feature_vector[index] = 1
        
        # Apply imputation
        imputed_vector = imputer.transform([feature_vector])[0]
        
        # Convert to DataFrame with valid feature names
        input_df = pd.DataFrame([imputed_vector], columns=symptoms)
        
        # Make prediction
        predicted_class = model.predict(input_df)[0]
        prediction = disease_names[predicted_class]
        
        # Get probabilities for all classes
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get top 3 diseases with their probabilities
        top_indices = prediction_proba.argsort()[-3:][::-1]
        top_diseases = [(disease_names[i], float(prediction_proba[i])) for i in top_indices]
        
        result = {
            'response': response,
            'prediction': prediction,
            'top_diseases': top_diseases,
            'confirmed_symptoms': confirmed_symptoms
        }
    else:
        result = {
            'response': response
        }
    
    return jsonify(result)

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    # Reset the chatbot conversation state
    chatbot.reset_conversation()
    return jsonify({'status': 'success', 'message': 'Chat reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)
