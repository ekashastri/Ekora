"""
Concrete implementations of AI and OCR providers.
"""
from app.services.interfaces import IOCRProvider, IAIProvider
import numpy as np
from typing import Optional
import cv2

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import google.generativeai as genai
    import os
    HAS_GENAI = True
    # We attempt to configure genai if there's a key
    if os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
except ImportError:
    HAS_GENAI = False

class TesseractOCRProvider(IOCRProvider):
    def extract_text(self, image: np.ndarray) -> str:
        if not HAS_TESSERACT:
            return "Pytesseract is not installed (Mock: Extracted Text)"
        if image is None:
            return "No image provided"
        try:
            # Pytesseract expects RGB image
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb = image
            text = pytesseract.image_to_string(rgb)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {e}"

class GeminiAIProvider(IAIProvider):
    def generate_diagram(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        if not HAS_GENAI:
            return "google.generativeai is not installed. (Mock: graph TD; A-->B;)"
            
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            contents = [prompt]
            if image is not None:
                # Convert cv2 image to PIL image for Gemini
                from PIL import Image
                if len(image.shape) == 3 and image.shape[2] == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb = image
                pil_img = Image.fromarray(rgb)
                contents.append(pil_img)
                
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            return f"AI Error: {e}"
