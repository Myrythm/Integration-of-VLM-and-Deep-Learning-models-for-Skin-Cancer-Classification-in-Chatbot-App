import os
from dotenv import load_dotenv
from flask_session import Session

# Load environment variables
load_dotenv()

# Flask configuration
def init_app(app):
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)

    # Configure upload folder
    UPLOAD_FOLDER = 'static/uploads/'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# File configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model labels
LABELS = [
    'Karsinoma Sel Basal',
    'Karsinoma Sel Skuamosa', 
    'Melanoma',
    'Nevus'
]

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 