from flask import Flask, request, jsonify
import cv2
import numpy as np
from paddleocr import PaddleOCR
import easyocr
import pytesseract
from PIL import Image
import time
import re
from typing import List, Any
import os
from sample import process_image_with_paddleocr
app = Flask(__name__)

def preprocess(img: np.ndarray) -> np.ndarray:
    """Convert BGR → Grayscale → Apply CLAHE (best for OCR on Indian IDs)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 3.0 is better than 2.0
    enhanced = clahe.apply(gray)
    return enhanced


def extract_text_from_paddle_result(result) -> str:
    if not result or not result[0]:
        return "(no text detected)"

    lines = []
    for page in result:           # usually only 1 page → result[0]
        for line in page:
            # line format: [bbox, (text, confidence)]
            if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                text = line[1][0].strip()
                conf = float(line[1][1])
                if conf > 0.5 and len(text) > 1:           # filter garbage
                    clean = re.sub(r'[^\w\s\.\,\'\-\/\:0-9]', '', text)
                    if clean:
                        lines.append(clean)
    
    return "\n".join(lines) if lines else "(no text)"

@app.route('/ocr', methods=['POST'])
def ocr_api():
    try:
        start = time.time()

        if 'image' not in request.files:
            return jsonify({"error": "No image part"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        """# READ FILE ONLY ONCE — THIS IS CRITICAL
        raw_bytes = file.read()
        if len(raw_bytes) == 0:
            return jsonify({"error": "Empty file"}), 400

        # Decode image
        img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        # Preprocess
        processed_gray = preprocess(img)                     # ← enhanced grayscale

        save_path = "images/processed.png"
        os.makedirs("images", exist_ok=True)
        cv2.imwrite(save_path, processed_gray)"""

        ""# ====================== PADDLEOCR ===================== 
        paddle_text = "(failed)"
        """pil_img = Image.open(save_path)
        img = np.array(pil_img)
        reader = PaddleOCR( lang='en')
            # result = reader.ocr(img, det=True, rec=True, cls=True)
        result = reader.ocr(img)"""
        result = process_image_with_paddleocr(file)
        print(result)
        print("PaddleOCR Result:")
        
        paddle_text = extract_text_from_paddle_result(result)
    except Exception as e:
        print(e)
    return jsonify({
        "paddleocr": paddle_text,
        "processing_time_ms": int((time.time() - start) * 1000),
        "status": "success"
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=False)