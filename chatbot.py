# chatbot.py

import nltk
import spacy
import re
import numpy as np
import pickle

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

class SymptomExtractor:
    def __init__(self, symptoms_list):
        """Initialize the symptom extractor with a list of valid symptoms"""
        self.symptoms = symptoms_list
        # Create a mapping of normalized symptoms to actual symptom names
        self.symptom_mapping = {}
        for symptom in symptoms_list:
            # Create normalized version (lowercase, no underscores)
            normalized = symptom.lower().replace('_', ' ')
            self.symptom_mapping[normalized] = symptom
            
            # Also add versions without spaces
            no_spaces = normalized.replace(' ', '')
            self.symptom_mapping[no_spaces] = symptom
            
        # Common symptom synonyms and variations
        self.symptom_synonyms = {
            'fever': ['high temperature', 'hot', 'burning up', 'temperature'],
            'headache': ['head pain', 'head ache', 'migraine'],
            'cough': ['coughing', 'hack', 'dry cough', 'wet cough'],
            'fatigue': ['tired', 'exhausted', 'no energy', 'weakness', 'lethargic'],
            'pain': ['ache', 'hurt', 'sore', 'discomfort'],
            'nausea': ['feeling sick', 'queasy', 'upset stomach'],
            'dizziness': ['vertigo', 'lightheaded', 'faint', 'spinning'],
            'vomiting': ['throwing up', 'puking', 'being sick'],
            'diarrhea': ['loose stool', 'watery stool', 'frequent bowel movements'],
            'chest pain': ['chest discomfort', 'chest pressure', 'chest tightness'],
            'shortness of breath': ['difficulty breathing', 'can\'t breathe', 'breathlessness', 'sob'],
            'sore throat': ['throat pain', 'painful throat', 'scratchy throat'],
            'runny nose': ['nasal discharge', 'drippy nose'],
            'congestion': ['stuffy nose', 'blocked nose', 'nasal congestion'],
            'rash': ['skin eruption', 'hives', 'skin irritation', 'spots'],
            'joint pain': ['arthralgia', 'painful joints', 'achy joints'],
            'muscle pain': ['myalgia', 'muscle ache', 'sore muscles'],
            'abdominal pain': ['stomach pain', 'tummy pain', 'belly pain', 'stomach ache'],
            'chills': ['shivering', 'feeling cold'],
            'swelling': ['edema', 'inflammation', 'puffy', 'bloated'],
        }
        
    def extract_symptoms(self, user_message):
        """Extract symptoms from a user message"""
        # Normalize the message
        message = user_message.lower()
        
        # Tokenize the message
        doc = nlp(message)
        
        # Extract potential symptoms
        extracted_symptoms = set()
        
        # Direct matching with symptom list
        for symptom_normalized, symptom_original in self.symptom_mapping.items():
            if symptom_normalized in message:
                extracted_symptoms.add(symptom_original)
        
        # Check for synonyms
        for key_term, synonyms in self.symptom_synonyms.items():
            if key_term in message or any(syn in message for syn in synonyms):
                # Find matching symptoms that contain this key term
                for symptom_normalized, symptom_original in self.symptom_mapping.items():
                    if key_term in symptom_normalized:
                        extracted_symptoms.add(symptom_original)
        
        # Use NLP to find medical entities
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "SYMPTOM"]:
                # Try to map the entity to a known symptom
                entity_text = ent.text.lower()
                for symptom_normalized, symptom_original in self.symptom_mapping.items():
                    if entity_text in symptom_normalized or symptom_normalized in entity_text:
                        extracted_symptoms.add(symptom_original)
        
        # Look for symptom patterns with body parts
        body_parts = ['head', 'chest', 'stomach', 'back', 'throat', 'eye', 'ear', 'nose', 'leg', 'arm', 'foot', 'hand', 'neck', 'joint']
        pain_terms = ['pain', 'ache', 'hurt', 'sore', 'discomfort']
        
        for part in body_parts:
            for term in pain_terms:
                pattern = f"{part} {term}"
                if pattern in message:
                    # Find matching symptoms
                    for symptom_normalized, symptom_original in self.symptom_mapping.items():
                        if part in symptom_normalized and term in symptom_normalized:
                            extracted_symptoms.add(symptom_original)
        
        return list(extracted_symptoms)

class ChatbotResponse:
    def __init__(self, model_data_path='disease_model.pkl'):
        """Initialize the chatbot with the disease prediction model"""
        # Load the trained model components
        with open(model_data_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.symptoms = self.model_data['symptoms']
        self.disease_names = self.model_data['disease_names']
        self.imputer = self.model_data['imputer']
        
        # Initialize the symptom extractor
        self.symptom_extractor = SymptomExtractor(self.symptoms)
        
        # Keep track of the conversation state
        self.reset_conversation()
    
    def reset_conversation(self):
        """Reset the conversation state"""
        self.extracted_symptoms = []
        self.confirmed_symptoms = []
        self.current_state = "greeting"
    
    def process_message(self, user_message):
        """Process a user message and return a response"""
        # Extract symptoms from the message
        new_symptoms = self.symptom_extractor.extract_symptoms(user_message)
        
        # Add to extracted symptoms if not already present
        for symptom in new_symptoms:
            if symptom not in self.extracted_symptoms:
                self.extracted_symptoms.append(symptom)
        
        # Determine the response based on the current state
        if self.current_state == "greeting":
            if len(new_symptoms) > 0:
                self.current_state = "confirming_symptoms"
                return self._confirm_symptoms()
            else:
                self.current_state = "asking_symptoms"
                return "Hello! I'm here to help identify possible health conditions based on your symptoms. Could you describe what symptoms you're experiencing?"
        
        elif self.current_state == "asking_symptoms":
            if len(new_symptoms) > 0:
                self.current_state = "confirming_symptoms"
                return self._confirm_symptoms()
            else:
                return "I couldn't identify any specific symptoms from your message. Could you please describe your symptoms more clearly? For example, 'I have a headache and fever.'"
        
        elif self.current_state == "confirming_symptoms":
            # Check if user confirmed or denied symptoms
            if any(word in user_message.lower() for word in ["yes", "correct", "right", "yeah", "yep", "confirm"]):
                # Add all extracted symptoms to confirmed
                for symptom in self.extracted_symptoms:
                    if symptom not in self.confirmed_symptoms:
                        self.confirmed_symptoms.append(symptom)
                self.extracted_symptoms = []
                
                if len(self.confirmed_symptoms) >= 3:
                    self.current_state = "predicting"
                    return self._make_prediction()
                else:
                    self.current_state = "asking_more_symptoms"
                    return "Thank you for confirming. Do you have any other symptoms you'd like to mention?"
            
            elif any(word in user_message.lower() for word in ["no", "incorrect", "wrong", "nope", "not"]):
                self.extracted_symptoms = []
                self.current_state = "asking_symptoms"
                return "I apologize for the misunderstanding. Could you please describe your symptoms again?"
            
            else:
                # Extract more symptoms from this message
                if len(new_symptoms) > 0:
                    return self._confirm_symptoms()
                else:
                    return "I'm not sure if you're confirming these symptoms. Please respond with 'yes' if these symptoms are correct, or 'no' if they're not."
        
        elif self.current_state == "asking_more_symptoms":
            if len(new_symptoms) > 0:
                self.current_state = "confirming_symptoms"
                return self._confirm_symptoms()
            elif any(word in user_message.lower() for word in ["no", "that's all", "nothing else", "done", "finished"]):
                self.current_state = "predicting"
                return self._make_prediction()
            else:
                return "Do you have any other symptoms? If not, please say 'no' and I'll analyze the symptoms you've mentioned."
        
        elif self.current_state == "predicting":
            # Reset for a new conversation
            response = "I've already provided a prediction based on your symptoms. Would you like to start a new consultation?"
            if any(word in user_message.lower() for word in ["yes", "new", "start over", "reset"]):
                self.reset_conversation()
                response = "Let's start over. Please describe your symptoms."
            return response
        
        # Default response
        return "I'm here to help identify possible health conditions. Please describe your symptoms."
    
    def _confirm_symptoms(self):
        """Generate a confirmation message for extracted symptoms"""
        if not self.extracted_symptoms:
            self.current_state = "asking_symptoms"
            return "I couldn't identify any specific symptoms. Could you please describe what you're experiencing?"
        
        # Format the symptoms for display
        symptom_list = ", ".join([s.replace('_', ' ').title() for s in self.extracted_symptoms])
        
        return f"I identified these symptoms: {symptom_list}. Is this correct?"
    
    def _make_prediction(self):
        """Make a prediction based on confirmed symptoms"""
        if not self.confirmed_symptoms:
            self.current_state = "asking_symptoms"
            return "I don't have any confirmed symptoms to make a prediction. Could you please describe your symptoms?"
        
        # Create feature vector (all 0s initially)
        feature_vector = np.zeros(len(self.symptoms))
        
        # Set confirmed symptoms to 1
        for symptom in self.confirmed_symptoms:
            if symptom in self.symptoms:
                index = self.symptoms.index(symptom)
                feature_vector[index] = 1
        
        # Apply imputation
        imputed_vector = self.imputer.transform([feature_vector])[0]
        
        # Convert to DataFrame with valid feature names
        import pandas as pd
        input_df = pd.DataFrame([imputed_vector], columns=self.symptoms)
        
        # Make prediction
        predicted_class = self.model.predict(input_df)[0]
        prediction = self.disease_names[predicted_class]
        
        # Get probabilities for all classes
        prediction_proba = self.model.predict_proba(input_df)[0]
        
        # Get top 3 diseases with their probabilities
        top_indices = prediction_proba.argsort()[-3:][::-1]
        top_diseases = [(self.disease_names[i], float(prediction_proba[i])) for i in top_indices]
        
        # Format the response
        response = f"Based on the symptoms you've described, the most likely condition is: {prediction}\n\n"
        response += "Top 3 possible conditions:\n"
        
        for i, (disease, prob) in enumerate(top_diseases):
            percentage = prob * 100
            response += f"{i+1}. {disease} ({percentage:.1f}%)\n"
        
        response += "\nPlease note that this is not a medical diagnosis. Consult with a healthcare professional for proper medical advice."
        
        return response
