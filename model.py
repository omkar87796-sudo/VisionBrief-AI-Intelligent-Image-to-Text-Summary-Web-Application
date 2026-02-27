"""
Image Summary Generator
Using BLIP (Bootstrapped Language-Image Pre-training) model from HuggingFace
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
from io import BytesIO

class ImageSummaryGenerator:
    def __init__(self):
        print("Loading BLIP model... (first time may take a few minutes)")
        
        # Load BLIP model - good for image captioning + summary
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded! Using device: {self.device}")

    def generate_summary(self, image: Image.Image, max_length: int = 200) -> str:
        """
        Generate a detailed summary/description from an image.
        
        Args:
            image: PIL Image object
            max_length: Maximum length of generated summary
            
        Returns:
            Generated summary text
        """
        # Convert image to RGB (in case it's RGBA or grayscale)
        image = image.convert("RGB")
        
        # Process the image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate summary using beam search for better quality
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,          # Beam search for better output
                min_length=30,        # Minimum summary length
                repetition_penalty=1.5,
                length_penalty=1.0,
            )
        
        # Decode the output tokens to text
        summary = self.processor.decode(output[0], skip_special_tokens=True)
        return summary

    def generate_from_url(self, url: str) -> str:
        """Generate summary from an image URL."""
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        return self.generate_summary(image)

    def generate_from_path(self, path: str) -> str:
        """Generate summary from a local image file path."""
        image = Image.open(path)
        return self.generate_summary(image)


# ---- Quick Test ----
if __name__ == "__main__":
    generator = ImageSummaryGenerator()
    
    # Test with a sample image URL
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    print("\nGenerating summary for test image...")
    summary = generator.generate_from_url(test_url)
    print(f"\n📋 Summary: {summary}")
