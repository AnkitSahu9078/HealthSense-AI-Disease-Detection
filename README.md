# HealthSense AI - Disease Detection System

HealthSense AI is a Flask-based web application designed to predict potential diseases based on user-inputted symptoms. It offers two modes of interaction: a manual symptom selection interface and an intelligent chatbot assistant.

## Screenshot of HealthSense AI

![Manual_Symptom_Selection](images/Screenshot 2025-05-17 013704.png)

## Chatbot Interface

![Chatbot](images/Screenshot 2025-05-17 013810.png)

## Technologies Used

- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3
- **Machine Learning Model:** Scikit-learn (RandomForest), NLTK, spaCy
- **Database:** SQLite3
- **Environment Management:** python-dotenv
- **Version Control:** Git & GitHub

## Project Structure

```
HealthSense-AI-Disease-Detection/
├── app.py
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
├── templates/
│   └── index.html
├── ml_model/
│   └── model.pkl
├── .env
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Running the Application

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/HealthSense-AI-Disease-Detection.git
   cd HealthSense-AI-Disease-Detection
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   # For Windows
   venv\Scripts\activate
   # For Linux/macOS
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**

   ```bash
   python app.py
   ```

5. **Open your browser and go to:**
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

Thank You
