from flask import session
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_chat_response(user_message, detection_result=None):
    """Get response from ChatGPT with conversation history"""
    try:
        # Initialize or get conversation history from session
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        # Initialize or get detection history from session
        if 'detection_history' not in session:
            session['detection_history'] = []
        
        # If there's a new detection result, add it to detection history
        if detection_result and isinstance(detection_result, dict):
            session['detection_history'].append(detection_result)
            session.modified = True
            
            # Get detailed information about the detected condition
            label = detection_result.get('label', 'Tidak diketahui')
            confidence = detection_result.get('confidence', 0)
            validation_status = detection_result.get('validation_status', 'unknown')
            
            # Create detailed detection message
            detection_message = f"""
            The latest detection result using our AI model shows:
            - Type: {label}
            - Confidence level: {confidence * 100:.1f}%
            - Validation status: {validation_status}

            Additional information:
            - Supported classifications: Basal Cell Carcinoma, Squamous Cell Carcinoma, Melanoma, and Nevus
            - This result is an initial prediction and should be confirmed by a dermatologist

            """
            
            session['conversation_history'].append({
                "role": "system",
                "content": detection_message
            })
        
        # Prepare messages for API call
        messages = [
            {
                "role": "system", 
                "content": """
                Role: You are an expert medical assistant specializing in the early detection of skin cancer, 
                particularly basal cell carcinoma, squamous cell carcinoma, melanoma, and nevus.

                Guidelines:
                - Provide clear, evidence-based information about skin cancer symptoms, risk factors, prevention, and early detection.
                - Encourage users to seek professional medical evaluation for any concerns regarding skin lesions or changes in their skin.
                - Avoid diagnosing or recommending treatments—always direct users to qualified healthcare providers for further evaluation.
                - Politely decline to answer questions unrelated to skin cancer detection.
                - Remember and refer to previous detection results and conversation history when relevant.
                - When discussing detection results, always mention that they are from an AI model and should be confirmed by a dermatologist.
                - Provide context about the specific type of skin cancer detected (if any) and its general characteristics.
                
                Your goal is to educate users about skin cancer awareness while emphasizing the importance of professional medical assessment.
                """
            }
        ]
        
        # Add information about all previous detections if any
        if session.get('detection_history') and len(session['detection_history']) > 0:
            detection_summary = "Ringkasan hasil deteksi sebelumnya:\n"
            for idx, detection in enumerate(session['detection_history'], 1):
                label = detection.get('label', 'Tidak diketahui')
                confidence = detection.get('confidence', 0)
                validation_status = detection.get('validation_status', 'unknown')
                detection_summary += f"{idx}. Jenis: {label}, Kepercayaan: {confidence * 100:.1f}%, Status: {validation_status}\n"
            
            messages.append({
                "role": "system",
                "content": detection_summary
            })
        
        # Add conversation history
        messages.extend(session['conversation_history'])
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response from GPT-4o
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500,
                temperature=0.5
            )
            
            if not response or not response.choices or not response.choices[0].message:
                raise Exception("Invalid response from GPT-4o")
                
            # Get assistant's response
            assistant_response = response.choices[0].message.content.strip()
            
            if not assistant_response:
                raise Exception("Empty response from GPT-4o")
            
            # Update conversation history
            session['conversation_history'].append({
                "role": "user",
                "content": user_message
            })
            session['conversation_history'].append({
                "role": "assistant",
                "content": assistant_response
            })
            
            # Keep only last 10 messages to prevent session from getting too large
            if len(session['conversation_history']) > 10:
                session['conversation_history'] = session['conversation_history'][-10:]
            
            # Ensure session changes are saved
            session.modified = True
            
            return assistant_response
            
        except Exception as api_error:
            print(f"API Error: {str(api_error)}")
            # Return a fallback response if API call fails
            return "Maaf, saya mengalami kesalahan dalam memproses permintaan Anda. Silakan coba lagi atau hubungi administrator."
        
    except Exception as e:
        print(f"Error in get_chat_response: {str(e)}")
        return "Maaf, terjadi kesalahan dalam sistem. Silakan coba lagi nanti." 