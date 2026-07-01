import os
import base64
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load secret environment API tokens locally
load_dotenv()

# Robust absolute path calculation for Vercel Serverless environment
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.abspath(os.path.join(base_dir, '..', 'templates'))

app = Flask(__name__, template_folder=template_dir)

# Initialize Groq Cloud Engine safely
api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

@app.route('/')
def home():
    """Renders the main single-page interface application."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_substance():
    """Handles Mode 1: Known Substance Risk Analyzer functionality."""
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API key missing in environment backend config."})
    
    payload = request.get_json() or {}
    substance = payload.get('substance', 'Unknown')
    duration = payload.get('duration', 'Unknown')
    contact_method = payload.get('contact_method', 'Unknown')

    system_prompt = (
        "You are an educational first-aid assistant specializing in all kinds of poisonous substances and drugs. "
        "Provide immediate non-medical guidance based on the given hazard parameters. "
        "Structure your response clearly with headers: 1) Immediate Action Needed, "
        "2) Common Symptoms/Side Effects to monitor over 24 hours, and 3) Risk Level Assessment (High/Low)."
    )
    
    user_message = f"Substance: {substance}\nExposure Duration: {duration}\nContact Method: {contact_method}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        ai_response = chat_completion.choices[0].message.content
        return jsonify({"success": True, "data": ai_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/identify', methods=['POST'])
def identify_symptoms():
    """Handles Mode 2: Symptom Identifier Mode functionality."""
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API key missing in environment backend config."})
    
    payload = request.get_json() or {}
    symptoms = payload.get('symptoms', 'None reported')
    context = payload.get('context', 'Unknown environment')

    system_prompt = (
        "You are an educational first-aid safety analyzer. Review the reported physical human symptoms "
        "and corresponding environment activity context. Suggest 2 or 3 common household chemicals, "
        "plants, or environmental hazards that match this description. Conclude by prompting the user "
        "to check safely if these specific items are present nearby. Keep it structural and clear."
    )
    
    user_message = f"Symptoms experienced: {symptoms}\nEnvironment Context: {context}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        ai_response = chat_completion.choices[0].message.content
        return jsonify({"success": True, "data": ai_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/scan', methods=['POST'])
def scan_label_image():
    """Handles Advanced Phase C: Vision Processing for chemical label photos."""
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API engine client uninitialized."})
    
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file detected in form submission request."})
        
    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"success": False, "error": "Selected filename is empty."})
        
    try:
        raw_bytes = uploaded_file.read()
        base64_encoded = base64.b64encode(raw_bytes).decode('utf-8')
        
        mime_type = "image/jpeg"
        if uploaded_file.filename.lower().endswith('.png'):
            mime_type = "image/png"
            
        vision_system_instruction = (
            "You are an emergency educational first-aid scanner. Examine the provided chemical label or ingredients text image. "
            "1) Identify the primary chemical compounds present. "
            "2) Highlight if any ingredients pose hazardous risks. "
            "3) Outline high-level non-medical first-aid parameters if skin, eye, or ingestion exposure occurs."
        )
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_system_instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_encoded}"
                            }
                        }
                    ]
                }
            ],
            model="llama-3.2-11b-vision-preview",
            temperature=0.15
        )
        
        vision_result = chat_completion.choices[0].message.content
        return jsonify({"success": True, "data": vision_result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# NEW FEATURE: CLINICAL LAB REPORT PATTERN ANALYZER ROUTE
@app.route('/api/lab_analyze', methods=['POST'])
def analyze_lab_reports():
    """Analyzes physiological panel values across core biological organ systems."""
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API engine client uninitialized."})
        
    payload = request.get_json() or {}
    toxin = payload.get('toxin', '').strip()
    amount = payload.get('amount', '').strip()
    lab_values = payload.get('lab_values', '').strip()
    
    # STUDENT CUSTOMIZATION ZONE: Add additional backend verification parameters here if desired!
    # Quick structural check for negative numbers or obviously corrupted input fields
    if "-" in lab_values or "-" in amount:
        return jsonify({
            "success": True, 
            "data": "⚠️ CRITICAL EVALUATION ERROR\n\n[Confidence Score: 0%]\nReason: Invalid lab metrics provided. Lab results and exposure measurements cannot contain negative values. Please re-enter your official lab report statistics."
        })

    system_prompt = (
        "You are an educational physiological safety assistant. Analyze the user's submitted laboratory values "
        "against the suspected toxin exposure. Look for metabolic or stress patterns across major organ systems "
        "(such as Liver, Kidneys, Lungs, Heart, or Nervous System).\n\n"
        "STRICT COMPLIANCE RULES:\n"
        "1. DO NOT state definitively that the toxin caused these specific organ irregularities, as pre-existing diseases, "
        "chronic conditions, or baseline health histories are unverified.\n"
        "2. You must compute and explicitly output a 'Confidence Score' out of 100% at the very top of your response using these guidelines:\n"
        "   - If the toxin is completely unknown, fictional, or unrecognizable to toxicological literature, Confidence Score is 0%.\n"
        "   - If the user explicitly stated that the exposure amount is missing or if the amount field is completely unprovided/empty, reduce the confidence score significantly (assign no higher than 40-50%).\n"
        "   - If the lab values display typical physiological patterns correlated with a known exposure and all inputs are complete, assign a high confidence score (80-95%)."
    )
    
    user_message = (
        f"Suspected Toxin: {toxin if toxin else 'Not Provided'}\n"
        f"Exposure Amount: {amount if amount else 'NOT PROVIDED / MISSING'}\n"
        f"Submitted Lab Metrics: {lab_values}"
    )
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        ai_response = chat_completion.choices[0].message.content
        return jsonify({"success": True, "data": ai_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# NEW FEATURE: PHARMACEUTICAL OVERDOSE RESPONDER
@app.route('/api/overdose', methods=['POST'])
def analyze_overdose():
    """Handles evaluation of emergency pharmaceutical or recreational drug overdose metrics."""
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API engine client uninitialized."})
        
    payload = request.get_json() or {}
    drug = payload.get('drug', '').strip()
    normal_amount = payload.get('normal_amount', '').strip()
    schedule = payload.get('schedule', '').strip()
    other_meds = payload.get('other_meds', '').strip()
    overdose_amount = payload.get('overdose_amount', '').strip()

    system_prompt = (
        "You are an educational crisis response assistant specializing in pharmacology and toxic drug exposures. "
        "Examine the user's submitted input details regarding a suspected overdose situation. "
        "Provide immediate, non-medical educational directives detailing what to do next in this exact scenario. "
        "Organize your findings with clear, scannable structures including: 1) Critical First Steps, 2) Signs of Toxicity "
        "to Alert Emergency Responders, and 3) Potential Contraindications based on provided medication contexts."
    )
    
    user_message = (
        f"Suspected Medication/Drug: {drug if drug else 'Unknown'}\n"
        f"Normal Prescribed Dosage Amount: {normal_amount if normal_amount else 'Not Provided/Unsure'}\n"
        f"Prescribed Timing/Schedule: {schedule if schedule else 'Not Provided/Unsure'}\n"
        f"Other Concurrent Medications Taken: {other_meds if other_meds else 'None Reported'}\n"
        f"Estimated Overdosed Amount: {overdose_amount if overdose_amount else 'Unknown Quantity'}"
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        ai_response = chat_completion.choices[0].message.content
        return jsonify({"success": True, "data": ai_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# NEW FEATURE: ECOLOGICAL & ANATOMICAL IMPACT ANALYZER
@app.route('/api/eco_health', methods=['POST'])
def analyze_eco_health_impact():
    """Evaluates cross-boundary chemical risks targeting environmental or human frameworks."""
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API engine client uninitialized."})
        
    payload = request.get_json() or {}
    chemical = payload.get('chemical', '').strip()
    impact_target = payload.get('impact_target', 'human_body').strip()

    if impact_target == "ecosystem":
        system_prompt = (
            "You are an environmental hazard and ecological safety scientist. Analyze the chemical or compound provided. "
            "Detail exactly how this substance disrupts bio-networks across two specific domains:\n"
            "1) Land Ecosystems (soil composition, plant life, terrestrial wildlife, food chain accumulation).\n"
            "2) Sea Ecosystems (marine organisms, aquatic toxicity, water quality, oxygen depletion).\n"
            "Keep the response structural and informative."
        )
    else:
        system_prompt = (
            "You are a human physiological toxicology assistant. Analyze the chemical or compound provided. "
            "Detail exactly how it impacts human health by highlighting:\n"
            "1) Directly Affected Organs and physiological body systems.\n"
            "2) Severity & Mortality Risk: Clearly address whether exposure can be fatal or result in permanent damage (whether they will die or not).\n"
            "Keep the response structural and informative."
        )
        
    user_message = f"Chemical/Substance Name: {chemical}\nRequested Target Scope: {impact_target}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        ai_response = chat_completion.choices[0].message.content
        return jsonify({"success": True, "data": ai_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)