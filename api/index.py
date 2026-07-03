import os
import base64
import logging
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Set up professional logging configurations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.abspath(os.path.join(base_dir, '..', 'templates'))

app = Flask(__name__, template_folder=template_dir)

api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_substance():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API key missing configuration."})
    
    payload = request.get_json() or {}
    substance = payload.get('substance', 'Unknown')
    duration = payload.get('duration', 'Unknown')
    contact_method = payload.get('contact_method', 'Unknown')

    system_prompt = (
        "You are an educational first-aid assistant specializing in chemical hazards and toxic substances.\n\n"
        "Provide guidance using these precise headers:\n"
        "**Immediate Action Needed**\n"
        "**Common Symptoms to Monitor**\n"
        "**Risk Level Assessment**\n\n"
        "Inside the Risk Level Assessment section, you must include a line stating exactly the risk level using one of these formats:\n"
        "- Risk Level: High\n"
        "- Risk Level: Medium\n"
        "- Risk Level: Low"
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
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/describe_unknown', methods=['POST'])
def describe_unknown_substance():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API key missing configuration."})
    
    payload = request.get_json() or {}
    description = payload.get('description', 'Unknown profile')
    duration = payload.get('duration', 'Unknown')
    contact_method = payload.get('contact_method', 'Unknown')
    symptoms = payload.get('symptoms', 'None reported')

    system_prompt = (
        "You are an educational first-aid assistant specializing in unknown household hazard assessments.\n\n"
        "Analyze the visual descriptor details, exposure time metrics, contact pathway, and physical symptoms to "
        "provide a list of likely substance identity suggestions along with tactical safe steps for each hypothetical matches.\n\n"
        "Organize your evaluation using these precise major headers:\n"
        "**Potential Substance Suggestions**\n"
        "**Recommended Next Steps for Each Suggestion**\n"
        "**Risk Level Assessment**\n\n"
        "Inside the Risk Level Assessment section, include a line stating exactly the risk level using one of these formats:\n"
        "- Risk Level: High\n"
        "- Risk Level: Medium\n"
        "- Risk Level: Low"
    )
    user_message = f"Description: {description}\nExposure Duration: {duration}\nContact Method: {contact_method}\nSymptoms: {symptoms}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/identify', methods=['POST'])
def identify_symptoms():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API key missing configuration."})
    
    payload = request.get_json() or {}
    symptoms = payload.get('symptoms', 'None reported')
    context = payload.get('context', 'Unknown environment')

    system_prompt = (
        "You are an educational first-aid safety analyzer. Review the reported reactions and context.\n\n"
        "Organize your evaluation using these specific headers:\n"
        "**Primary Household Hazard Suspects**\n"
        "**Environmental Safety Context Evaluation**\n"
        "**Risk Level Assessment**"
    )
    user_message = f"Symptoms: {symptoms}\nContext: {context}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/scan', methods=['POST'])
def scan_label_image():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API client uninitialized."})
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file detected."})
        
    uploaded_file = request.files['image']
    try:
        raw_bytes = uploaded_file.read()
        base64_encoded = base64.b64encode(raw_bytes).decode('utf-8')
        mime_type = "image/png" if uploaded_file.filename.lower().endswith('.png') else "image/jpeg"
            
        vision_instruction = (
            "Examine this chemical label image. Structure your output text explicitly using these clean headers:\n"
            "**Identified Chemical Compounds**\n"
            "**Hazardous Ingredient Warnings**\n"
            "**Risk Level Assessment**"
        )
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_instruction},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_encoded}"}}
                    ]
                }
            ],
            model="llama-3.2-11b-vision-preview",
            temperature=0.15
        )
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/lab_analyze', methods=['POST'])
def analyze_lab_reports():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API client uninitialized."})
        
    payload = request.get_json() or {}
    toxin = payload.get('toxin', '').strip()
    amount = payload.get('amount', '').strip()
    lab_values = payload.get('lab_values', '').strip()

    system_prompt = (
        "Analyze the laboratory values against the suspected toxin exposure.\n\n"
        "Break down your parameters using these specific major headers:\n"
        "**Confidence Score Assessment Metrics**\n"
        "**Biomarker Organ Stress Footprint**\n"
        "**Risk Level Assessment**"
    )
    user_message = f"Suspected Toxin: {toxin}\nExposure Amount: {amount}\nMetrics: {lab_values}"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/overdose', methods=['POST'])
def analyze_overdose():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API client uninitialized."})
        
    payload = request.get_json() or {}
    drug = payload.get('drug', '').strip()
    normal_amount = payload.get('normal_amount', '').strip()
    schedule = payload.get('schedule', '').strip()
    other_meds = payload.get('other_meds', '').strip()
    overdose_amount = payload.get('overdose_amount', '').strip()

    system_prompt = (
        "You are an educational pharmacology crisis response assistant.\n\n"
        "Organize your evaluation with these scannable structures:\n"
        "**Critical First Steps**\n"
        "**Signs of Toxicity to Alert Emergency Responders**\n"
        "**Potential Contraindications based on Provided Medication Contexts**\n"
        "**Risk Level Assessment**\n"
        "**Additional Information**"
    )
    user_message = f"Drug: {drug}\nNormal Amount: {normal_amount}\nSchedule: {schedule}\nMeds: {other_meds}\nOverdosed Amount: {overdose_amount}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/eco_health', methods=['POST'])
def analyze_eco_health_impact():
    if not groq_client:
        return jsonify({"success": False, "error": "Groq API client uninitialized."})
        
    payload = request.get_json() or {}
    chemical = payload.get('chemical', '').strip()
    impact_target = payload.get('impact_target', 'human_body').strip()

    if impact_target == "ecosystem":
        system_prompt = (
            "Analyze the chemical compound provided. Detail disruptions across these domains:\n"
            "**Land and Aquatic Ecosystem Disruption**\n"
            "**Risk Level Assessment**"
        )
    else:
        system_prompt = (
            "Analyze the chemical compound provided. Detail disruptions across these domains:\n"
            "**Systemic Internal Body Organ Hazards**\n"
            "**Risk Level Assessment**"
        )
    user_message = f"Chemical: {chemical}\nTarget: {impact_target}"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )
        return jsonify({"success": True, "data": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)