import os
import json
import hashlib
import base64
import cv2
import numpy as np
import pytesseract
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# --- Configuration ---
app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///med_history.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Note: pytesseract relies on the 'tesseract' binary being installed on your system.
# On Linux (Arch): sudo pacman -S tesseract tesseract-data-eng

# --- Database Models ---
class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_hash = db.Column(db.String(64), nullable=False)
    raw_ocr_text = db.Column(db.Text, nullable=True)
    identified_name = db.Column(db.String(255), nullable=True)
    json_result = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- Helper Functions ---
def process_image_for_ocr(image_bytes):
    """
    Pre-processes image to improve Tesseract accuracy.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return img, thresh

def analyze_with_gemini(image_base64, ocr_text):
    """
    Uses LangChain and Gemini to analyze the image and OCR text.
    """
    # UPDATED: Using gemini-2.5-flash as requested
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt_text = f"""
    You are an expert pharmacist AI designed to identify damaged medicine wrappers.
    
    1. I have performed Tesseract OCR on this damaged wrapper and found this partial text: "{ocr_text}".
    2. Please look at the attached image carefully. Combine the visual cues (colors, logos, remaining letters) with the OCR text to identify the medicine.
    
    If the wrapper is too damaged to be 100% sure, provide the most likely candidate and a warning.

    You must output a strictly valid JSON object with the following structure (do not include markdown formatting like ```json):
    {{
        "medicine_name": "Standard Name",
        "confidence_score": "High/Medium/Low",
        "details": {{
            "english": {{
                "usage": "Usage instructions...",
                "dosage": "Dosage info...",
                "warnings": "..."
            }},
            "hindi": {{
                "usage": "Hindi translation of usage...",
                "dosage": "Hindi translation of dosage...",
                "warnings": "..."
            }},
            "gujarati": {{
                "usage": "Gujarati translation of usage...",
                "dosage": "Gujarati translation of dosage...",
                "warnings": "..."
            }}
        }},
        "company_info": {{
            "name": "Manufacturer Name",
            "certification": "FDA/ISI/WHO-GMP status (Estimated based on general knowledge of this brand)"
        }}
    }}
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
        ]
    )

    response = llm.invoke([message])
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # 1. Read Image
        img_bytes = file.read()
        
        # 2. Tesseract OCR
        original_img, processed_img = process_image_for_ocr(img_bytes)
        ocr_text = pytesseract.image_to_string(processed_img, config='--psm 6')
        ocr_text = " ".join(ocr_text.split())
        print(f"DEBUG: Tesseract Found: {ocr_text}")

        # 3. Prepare for Gemini
        pil_img = Image.open(BytesIO(img_bytes))
        
        # Convert RGBA to RGB (Fix for PNGs)
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
            
        pil_img.thumbnail((1024, 1024)) 
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 4. Gemini Analysis
        ai_result = analyze_with_gemini(img_base64, ocr_text)

        # 5. Save to DB
        new_scan = ScanHistory(
            image_hash=hashlib.md5(img_bytes).hexdigest(),
            raw_ocr_text=ocr_text,
            identified_name=ai_result.get('medicine_name', 'Unknown'),
            json_result=json.dumps(ai_result)
        )
        db.session.add(new_scan)
        db.session.commit()

        return jsonify(ai_result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    # Fetch last 10 scans
    scans = ScanHistory.query.order_by(ScanHistory.timestamp.desc()).limit(10).all()
    history_data = []
    
    for scan in scans:
        try:
            # Parse the stored string back into a JSON object
            full_data = json.loads(scan.json_result) if scan.json_result else {}
        except json.JSONDecodeError:
            full_data = {}

        history_data.append({
            "id": scan.id,
            "name": scan.identified_name or "Unknown",
            "time": scan.timestamp.strftime("%d %b, %H:%M"), # Format: 28 Nov, 14:30
            "full_data": full_data 
        })
    
    return jsonify(history_data)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, port=5000)