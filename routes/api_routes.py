import os
from flask import Blueprint, request, jsonify, current_app, session
from tensorflow.keras.models import load_model
from utils.image_processing import allowed_file, prepare_image
from utils.validation import validate_skin_image
from utils.chat import get_chat_response
from config import LABELS
import numpy as np

api = Blueprint('api', __name__)

# Load TF model
try:
    model = load_model('./model/skinCancer.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@api.route('/detect', methods=['POST'])
def detect():
    """Handle image detection request"""
    # Check if model is available
    if model is None:
        return jsonify({'error': 'Model tidak tersedia'}), 500

    # Check if image was uploaded   
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400

    file = request.files['image']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    # Process valid image file
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Validate image using OpenAI Vision
            validation_result = validate_skin_image(filepath)
            print(f"Validation result: {validation_result}")  # Debug print
            
            # Process if validation result is 'valid' or 'uncertain'
            if validation_result in ['valid', 'uncertain']:
                # Prepare image for prediction
                img_array = prepare_image(filepath)
                if img_array is None:
                    raise ValueError("Gagal memproses gambar")

                # Make prediction
                predictions = model.predict(img_array)
                confidence = float(np.max(predictions))
                predicted_class = LABELS[np.argmax(predictions)]

                # Create detection result
                detection = {
                    'label': predicted_class,
                    'confidence': confidence,
                    'validation_status': validation_result
                }
                
                # Store detection result in session
                if 'detection_history' not in session:
                    session['detection_history'] = []
                session['detection_history'].append(detection)
                session.modified = True
                
                # Add validation status to response
                result = {
                    'detections': [detection]
                }

                # Clean up - remove temporary file
                os.remove(filepath)
                
                return jsonify(result)
            else:
                os.remove(filepath)
                return jsonify({
                    'error': 'Gambar yang diunggah bukan gambar kanker kulit',
                    'validation_status': validation_result
                }), 400

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error saat memproses gambar: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Format file tidak didukung'}), 400

@api.route('/get_response', methods=['POST'])
def get_response():
    """Handle chat response request"""
    try:
        data = request.json
        if not data:
            return jsonify({"reply": "Permintaan tidak valid"}), 400
            
        user_message = data.get('message')
        if not user_message:
            return jsonify({"reply": "Pesan tidak boleh kosong"}), 400
            
        detection_result = data.get('detection_result')
        
        # If detection_result is provided, store it as a single detection object
        if detection_result and isinstance(detection_result, dict) and 'detections' in detection_result:
            detection_obj = detection_result['detections'][0] if detection_result['detections'] else None
        else:
            detection_obj = None
        
        chat_response = get_chat_response(user_message, detection_obj)
        
        if not chat_response:
            return jsonify({"reply": "Maaf, tidak dapat memproses permintaan Anda saat ini"}), 500

        return jsonify({"reply": chat_response})

    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return jsonify({"reply": "Maaf, terjadi kesalahan dalam memproses permintaan Anda. Silakan coba lagi nanti."}), 500

@api.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation and detection history"""
    try:
        session['conversation_history'] = []
        session['detection_history'] = []
        session.modified = True
        return jsonify({"status": "success", "message": "Riwayat percakapan dan deteksi telah dihapus"})
    except Exception as e:
        print(f"Error clearing history: {str(e)}")
        return jsonify({"status": "error", "message": "Gagal menghapus riwayat"}), 500

@api.route('/get_detection_history', methods=['GET'])
def get_detection_history():
    """Return detection history from session"""
    try:
        detection_history = session.get('detection_history', [])
        return jsonify({"history": detection_history})
    except Exception as e:
        print(f"Error getting detection history: {str(e)}")
        return jsonify({"error": "Gagal mengambil riwayat deteksi"}), 500 