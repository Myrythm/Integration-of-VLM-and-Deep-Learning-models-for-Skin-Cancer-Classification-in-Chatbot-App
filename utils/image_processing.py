from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from config import ALLOWED_EXTENSIONS

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    """Prepare image for model prediction"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None 