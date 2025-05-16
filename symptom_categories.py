# symptom_categories.py

def categorize_symptoms(symptoms_list):
    """Categorize symptoms into logical groups"""
    
    # Define categories and their keywords
    categories = {
        "General": ["fatigue", "weakness", "malaise", "weight", "fever", "chills", "sweating", "dehydration"],
        "Head & Neurological": ["headache", "dizziness", "fainting", "consciousness", "seizures", "memory", "confusion", "disorientation", "speech", "migraine", "vertigo"],
        "Eyes": ["vision", "eye", "sight", "blind", "blurry", "yellowing", "pupils"],
        "Ears": ["ear", "hearing", "deaf", "tinnitus", "earache"],
        "Nose & Sinuses": ["nose", "nasal", "smell", "sneeze", "sinus", "congestion", "runny"],
        "Mouth & Throat": ["mouth", "throat", "tongue", "gums", "taste", "swallow", "sore throat", "hoarseness", "voice"],
        "Respiratory": ["breath", "cough", "wheeze", "chest tightness", "phlegm", "sputum", "lung", "respiratory", "asthma", "pneumonia", "bronchitis"],
        "Cardiovascular": ["heart", "chest pain", "palpitations", "pulse", "blood pressure", "hypertension", "circulation", "varicose", "edema"],
        "Digestive": ["stomach", "abdomen", "nausea", "vomit", "diarrhea", "constipation", "bowel", "intestine", "indigestion", "appetite", "flatulence", "bloating", "gastritis", "ulcer"],
        "Urinary": ["urine", "urination", "bladder", "kidney", "urinary", "incontinence"],
        "Reproductive": ["genital", "sexual", "menstrual", "vaginal", "erectile", "libido", "fertility", "pregnancy"],
        "Skin & Hair": ["skin", "rash", "itch", "hives", "acne", "eczema", "psoriasis", "hair", "nail", "sweat"],
        "Musculoskeletal": ["muscle", "joint", "bone", "back", "neck", "shoulder", "arm", "leg", "knee", "ankle", "wrist", "arthritis", "pain", "stiffness", "sprain", "strain"],
        "Psychological": ["anxiety", "depression", "stress", "mood", "sleep", "insomnia", "mental", "psychological", "psychiatric", "behavior", "emotion"],
        "Immune & Allergic": ["allergy", "immune", "infection", "inflammation", "swelling", "lymph", "autoimmune"],
        "Blood & Metabolic": ["blood", "anemia", "diabetes", "thyroid", "cholesterol", "metabolism", "vitamin", "mineral", "deficiency"]
    }
    
    # Categorize each symptom
    categorized_symptoms = {category: [] for category in categories}
    uncategorized = []
    
    for symptom in symptoms_list:
        # Convert to display format for matching
        display_symptom = symptom.replace('_', ' ').lower()
        
        # Try to find a category
        found_category = False
        for category, keywords in categories.items():
            if any(keyword in display_symptom for keyword in keywords):
                categorized_symptoms[category].append(symptom)
                found_category = True
                break
        
        # If no category found, add to uncategorized
        if not found_category:
            uncategorized.append(symptom)
    
    # Add uncategorized symptoms to "Other" category
    if uncategorized:
        categorized_symptoms["Other"] = uncategorized
    
    # Remove empty categories
    categorized_symptoms = {k: v for k, v in categorized_symptoms.items() if v}
    
    return categorized_symptoms
