import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, session
from torchvision import transforms
from PIL import Image
import google.generativeai as genai
import numpy as np
import mysql.connector 
import datetime
import io
import os

app = Flask(__name__)
app.secret_key = "medical_project_secret_key_2025"

GEMINI_API_KEY = "Your Api key"
genai.configure(api_key=GEMINI_API_KEY)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Your password', 
    'database': 'hospital_ai_db',
    
   
}

def get_db_connection():
    try:
     
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f" DATABASE ERROR: {err}")
        return None

def init_db():
    """Creates the database tables and default doctor if they don't exist."""
    print(" Checking Database System...")
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                password VARCHAR(50)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                age INT,
                diagnosis TEXT,
                date VARCHAR(50)
            )
        ''')
        
        cursor.execute("SELECT * FROM users WHERE username = 'doctor'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", ('doctor', 'admin123'))
            print("   -> Default Account Created: User='doctor', Pass='admin123'")
        
        conn.commit()
        cursor.close()
        conn.close()
        print(" Database System: ONLINE")
    else:
        print(" WARNING: Running WITHOUT Database. Login features will not work.")

init_db()
class MultiDiseaseModel(nn.Module):
    def __init__(self):
        super(MultiDiseaseModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.tabular_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 + 8, 64), nn.ReLU(), nn.Linear(64, 14), nn.Sigmoid())
    def forward(self, img, tab): 
        return self.classifier(torch.cat((self.features(img).view(img.size(0), -1), self.tabular_mlp(tab)), dim=1))

class CustomMedicalCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomMedicalCNN, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

print("\n-----------------------------------")
print(" INITIALIZING AI BRAINS...")

xray_model = MultiDiseaseModel()
XRAY_DISEASES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
if os.path.exists("multi_disease_model.pth"):
    try:
        xray_model.load_state_dict(torch.load("multi_disease_model.pth", map_location='cpu'))
        xray_model.eval()
        print(" X-Ray Model: READY")
    except: print(" X-Ray Model Corrupted")
else: print(" X-Ray Model Missing")

def load_specialist(name, primary_name, fallback_name):
    filename = primary_name if os.path.exists(primary_name) else fallback_name
    
    if not os.path.exists(filename):
        print(f" {name} Model: MISSING (Tried {primary_name} & {fallback_name})")
        return None, []
    
    try:
        checkpoint = torch.load(filename, map_location='cpu')
        model = CustomMedicalCNN(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        print(f" {name} Model: READY ({len(checkpoint['classes'])} classes)")
        return model, checkpoint['classes']
    except Exception as e:
        print(f" {name} Load Error: {e}")
        return None, []
mri_model, mri_classes = load_specialist("MRI", "mri_custom.pth", "mri_model.pth")
skin_model, skin_classes = load_specialist("Skin", "skin_custom.pth", "skin_model.pth")
ct_model, ct_classes = load_specialist("CT", "ct_custom.pth", "ct_model.pth")

print("-----------------------------------\n")

@app.route('/')
def home():
    is_doctor = 'user' in session
    return render_template('index.html', is_doctor=is_doctor)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = username
            return jsonify({'status': 'success'})

    if username == "doctor" and password == "admin123":
        session['user'] = "doctor"
        return jsonify({'status': 'success'})
        
    return jsonify({'status': 'fail'})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return jsonify({'status': 'success'})

@app.route('/save_patient', methods=['POST'])
def save_patient():
    if 'user' not in session: return jsonify({'error': 'Login Required'})
    data = request.json
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO patients (name, age, diagnosis, date) VALUES (%s, %s, %s, %s)",
                  (data['name'], data['age'], data['diagnosis'], datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        conn.close()
        return jsonify({'status': 'saved'})
    return jsonify({'error': 'DB Error'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        scan_type = request.form.get('scan_type')
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        img_t = transform(img).unsqueeze(0)

        results = {}
        high_risk = []

        if scan_type == 'xray':
            if not xray_model: return jsonify({'error': "X-Ray Model not loaded"})
            
            age = float(request.form['age'])
            gender = float(request.form['gender'])
            hr = float(request.form.get('hr', 72))
            temp_c = float(request.form['temp'])
            temp_f = (temp_c * 9/5) + 32
            
            tab_norm = [(age-46)/16, (gender-0.5)/0.5, (temp_f-99)/1.5, (hr-80)/10]
            tab_t = torch.tensor([tab_norm], dtype=torch.float32)

            with torch.no_grad(): preds = xray_model(img_t, tab_t)[0].numpy()
            for i, d in enumerate(XRAY_DISEASES):
                score = float(preds[i])
                results[d] = score
                if score > 0.3: high_risk.append(f"{d} ({int(score*100)}%)")

        else:
            active_model, active_classes = None, []
            if scan_type == 'mri': active_model, active_classes = mri_model, mri_classes
            elif scan_type == 'skin': active_model, active_classes = skin_model, skin_classes
            elif scan_type == 'ct': active_model, active_classes = ct_model, ct_classes
            
            if not active_model: return jsonify({'error': f"Model for {scan_type} not found/trained."})

            with torch.no_grad():
                outputs = active_model(img_t)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
            
            for i, d in enumerate(active_classes):
                score = float(probs[i])
                results[d] = score
                if score > 0.5: high_risk.append(f"{d} ({int(score*100)}%)")

        summary = f"{scan_type.upper()} Analysis: " + ", ".join(high_risk) if high_risk else "No significant findings."
        return jsonify({'results': results, 'summary': summary})

    except Exception as e: return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    context = data.get('context')
    user_msg = data.get('message')
    
    
    is_doctor = 'user' in session
    
    if is_doctor:
    
        system_instruction = """
        You are a **Senior Medical Consultant** speaking to a **Junior Doctor/Surgeon**.
        
        **TONE:** Professional, Clinical, Objective, Technical.
        **VOCABULARY:** Use terms like 'Prognosis', 'Etiology', 'Comorbidities', 'Contraindications', 'Therapeutic Modalities'.
        
        **YOUR TASK:**
        1. **Correlate Vitals:** Analyze how the Patient's Age, BP, and Sugar impact the Scan Diagnosis (e.g., "Hypertension complicates the anesthesia protocol for this surgery").
        2. **Surgical Strategy:** Suggest specific surgical approaches (e.g., "Consider Pterional Craniotomy").
        3. **Risk Stratification:** Assess the ASA Physical Status classification based on vitals.
        """
    else:
       
        system_instruction = """
        You are a **Kind, Empathetic Senior Doctor** speaking directly to the **Patient**.
        
        **TONE:** Reassuring, Simple, Clear, Educational.
        **VOCABULARY:** Avoid jargon. Use simple analogies (e.g., instead of 'Hypertension', say 'High Blood Pressure').
        
        **YOUR TASK:**
        1. **Explain the Report:** Tell them what the scan found in plain English.
        2. **Connect to Vitals:** Explain why their BP/Sugar matters (e.g., "Your high sugar might slow down healing, so we need to manage it").
        3. **Next Steps:** Give actionable advice (Diet, Sleep, Specialist to visit).
        4. **Reassurance:** Always end with a comforting note, but be honest.
        """

    prompt = f"""
    {system_instruction}
    
    **PATIENT CLINICAL DATA:**
    {context}
    
    **CURRENT QUESTION/TRIGGER:**
    "{user_msg}"
    
    **OUTPUT RULES:**
    - Use HTML <b>Bold</b> tags for key points.
    - Organize with bullet points.
    - If the user sends a greeting, give a brief medical summary based on the data provided.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'reply': f"System Error: {str(e)}"})
if __name__ == '__main__':
    app.run(debug=True, port=5000)
