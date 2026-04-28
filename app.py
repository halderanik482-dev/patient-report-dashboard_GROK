import streamlit as st
from groq import Groq
import json
from PyPDF2 import PdfReader
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MyHealth Insights", page_icon="🩺", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE GROQ API ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("🚨 System Error: GROQ_API_KEY not found in Secrets. Please configure the app settings.")
    st.stop()

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def analyze_report(content, file_type="text"):
    system_prompt = """
    You are an expert medical data assistant. Analyze the provided medical report.
    Identify all biomarkers (e.g., Vitamin D, Ferritin, Hemoglobin).
    Output the data STRICTLY as a JSON object with a single key "biomarkers" containing a list of dictionaries.
    Each dictionary must have exactly these keys: "biomarker", "value", "range", and "status".
    "status" must be "Low", "Normal", or "High" based on the range.
    Example: {"biomarkers": [{"biomarker": "Iron", "value": "85", "range": "60-170", "status": "Normal"}]}
    """
    
    try:
        if file_type == "text":
            # Use Llama 3 70B for standard PDF text extraction
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the medical report text:\n\n{content}"}
                ],
                model="llama3-70b-8192",
                response_format={"type": "json_object"}
            )
        else:
            # Use Llama 3.2 Vision for Image Uploads
            base64_image = base64.b64encode(content).decode('utf-8')
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the biomarkers from this medical report image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                model="llama-3.2-11b-vision-preview",
                response_format={"type": "json_object"}
            )
            
        # Parse the JSON response securely
        result_string = response.choices[0].message.content
        data = json.loads(result_string)
        return data.get("biomarkers", [])
        
    except Exception as e:
        st.error(f"Error communicating with Groq AI: {e}")
        return None

# --- MAIN UI ---
st.markdown("<div class='main-header'>🩺 MyHealth Insights</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Powered by Groq AI. Upload your report to instantly see what needs attention.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Report (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    if st.button("Analyze My Report", type="primary", use_container_width=True):
        with st.spinner("Analyzing your report at lightning speed..."):
            
            # 1. Process File
            if uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
                data = analyze_report(content, "text")
            else:
                # Pass raw bytes for image processing
                content = uploaded_file.getvalue()
                data = analyze_report(content, "image")
            
            # 2. Display Results Visually
            if data:
                st.markdown("---")
                st.subheader("Actionable Insights")
                
                # Filter for items outside normal range
                action_items = [item for item in data if item.get('status', 'Normal') != 'Normal']
                normal_items = [item for item in data if item.get('status', 'Normal') == 'Normal']
                
                if action_items:
                    st.error("🚨 **Areas Requiring Attention:**")
                    for item in action_items:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            col1.markdown(f"**{item.get('biomarker', 'Unknown')}**")
                            col2.metric("Your Result", item.get('value', 'N/A'))
                            col3.metric("Safe Range", item.get('range', 'N/A'))
                            st.write(f"*Status:* **{item.get('status', 'Check Report')}**")
                            st.divider()
                else:
                    st.success("🎉 All parameters found in the report are within the normal safe ranges!")
                
                if normal_items:
                    with st.expander("View Normal Parameters"):
                        for item in normal_items:
                            st.write(f"✅ **{item.get('biomarker')}**: {item.get('value')} (Range: {item.get('range')})")
