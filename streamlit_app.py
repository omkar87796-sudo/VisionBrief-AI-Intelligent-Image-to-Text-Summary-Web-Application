"""
VisionBrief AI - Image Summary Generator
Run locally: streamlit run streamlit_app.py
Deploy: Push to GitHub → connect to share.streamlit.io (FREE!)
"""

import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
from io import BytesIO

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="VisionBrief AI",
    page_icon="🖼️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }
    h1 { 
        background: linear-gradient(135deg, #7c6af7, #f77c8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    .summary-box {
        background: #13131a;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #e8e8f0;
        margin-top: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c6af7, #9a6af7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Load Model (cached so it loads only once) ────────────────
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


# ── Generate Summary ─────────────────────────────────────────
def generate_summary(image, processor, model, device, max_length=200):
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            min_length=30,
            repetition_penalty=1.5,
        )
    return processor.decode(output[0], skip_special_tokens=True)


# ── UI ───────────────────────────────────────────────────────
st.title("🖼️ VisionBrief AI")
st.markdown("##### Upload any image — our deep learning model will describe it for you")
st.divider()

# Load model with spinner
with st.spinner("Loading BLIP model... (first time takes ~1 min)"):
    processor, model, device = load_model()

st.success(f"✅ Model ready! Running on **{device.upper()}**")
st.divider()

# Input tabs
tab1, tab2 = st.tabs(["📁 Upload Image", "🔗 Image URL"])

image = None

with tab1:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with tab2:
    url = st.text_input("Paste image URL here...")
    if url:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_container_width=True)
        except:
            st.error("❌ Could not load image from URL. Please check the link.")

st.divider()

# Generate button
if st.button("✨ Generate Summary"):
    if image is None:
        st.warning("⚠️ Please upload an image or enter a URL first!")
    else:
        with st.spinner("Analyzing image with BLIP model..."):
            summary = generate_summary(image, processor, model, device)

        st.markdown("### 📋 Summary")
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
        st.divider()
        st.download_button(
            label="⬇️ Download Summary as .txt",
            data=summary,
            file_name="image_summary.txt",
            mime="text/plain"
        )
