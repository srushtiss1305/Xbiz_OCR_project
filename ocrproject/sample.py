from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

def process_image_with_paddleocr(image_path: str):
    # pil_img = Image.open(r"C:\Users\Sanjay\OneDrive\Desktop\flask-api\images\driving-licence.jpg")
    pil_img = Image.open(image_path)
    img = np.array(pil_img)
    reader = PaddleOCR( lang='en')
    # result = reader.ocr(img, det=True, rec=True, cls=True)
    result = reader.ocr(img)
    print(result)
    return result

result = process_image_with_paddleocr(r"C:\Users\Sanjay\OneDrive\Desktop\flask-api\images\driving-licence.jpg")