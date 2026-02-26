# 🖼️ Image Summary Generator

A deep learning model that generates text summaries from images using the **BLIP** model.

---

## 📁 Project Structure

```
image_summary_generator/
│
├── model.py          ← Core deep learning model (BLIP)
├── app.py            ← Flask web application
├── requirements.txt  ← Python dependencies
├── templates/
│   └── index.html    ← Web UI
└── README.md
```

---

## ⚙️ Setup & Installation

### Step 1 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First install will download PyTorch (~1-2 GB). Be patient!

### Step 3 — Run the app
```bash
python app.py
```

### Step 4 — Open in browser
```
http://localhost:5000
```

---

## 🧠 How It Works

1. **You upload an image** (or paste a URL)
2. The **BLIP model** (Salesforce/blip-image-captioning-large) processes it
3. It uses a **Vision Transformer** to encode visual features
4. A **language decoder** generates the summary text
5. **Beam search** is used for high-quality output

---

## 🔧 Test the Model Directly (Without Web UI)

```bash
python model.py
```

Or in Python:
```python
from model import ImageSummaryGenerator
from PIL import Image

generator = ImageSummaryGenerator()
image = Image.open("your_image.jpg")
summary = generator.generate_summary(image)
print(summary)
```

---

## 💡 Tips

- GPU (CUDA) will make it **much faster** — CPU works but is slower
- The model auto-downloads on first run (~1.8 GB)
- You can change `max_length` and `num_beams` in `model.py` to control output quality vs speed

---

## 🚀 Upgrade Ideas

- Use `BLIP-2` for even better summaries
- Add batch processing for multiple images
- Export summaries to PDF/CSV
- Add image OCR (text extraction) alongside summary
# VisionBrief-AI-Intelligent-Image-to-Text-Summary-Web-Application
