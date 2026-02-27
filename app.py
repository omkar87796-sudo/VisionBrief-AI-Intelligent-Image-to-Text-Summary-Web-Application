"""
Flask Web App for Image Summary Generator
Run: python app.py
Then open: http://localhost:5000
NO templates folder needed - HTML is embedded directly.
"""

from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import os
from model import ImageSummaryGenerator

app = Flask(__name__)
generator = ImageSummaryGenerator()

HTML_PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")) else "<h1>index.html not found</h1>"


@app.route("/")
def index():
    # Try to load from index.html in same folder, fallback to templates/
    base = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(base, "index.html"),
                 os.path.join(base, "templates", "index.html")]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
    return "<h1 style='color:red'>ERROR: index.html not found next to app.py or in templates/</h1>", 500


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream)
        elif request.json and "image_base64" in request.json:
            image_data = request.json["image_base64"].split(",")[-1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif request.json and "image_url" in request.json:
            import requests as req
            response = req.get(request.json["image_url"], timeout=10)
            image = Image.open(io.BytesIO(response.content))
        else:
            return jsonify({"error": "No image provided"}), 400

        summary = generator.generate_summary(image)
        return jsonify({"summary": summary, "status": "success"})

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    print("Starting Image Summary Generator Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
