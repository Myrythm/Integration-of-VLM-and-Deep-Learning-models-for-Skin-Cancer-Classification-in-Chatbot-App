import base64
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def validate_skin_image(image_path):
    """Validate if the image is a skin lesion using OpenAI Vision"""
    try:
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a dermatology imaging specialist trained to analyze skin lesions for signs of malignancy, 
                    including both cancerous (melanoma, basal cell carcinoma, squamous cell carcinoma) and benign (nevi, seborrheic keratosis) 
                    conditions, as well as other non-cancerous skin diseases (acne, eczema, psoriasis, warts, etc.)
                    
                    Based on the image provided, classify it according to these rules:
                    1. If the image clearly shows possible skin cancer (e.g., irregular borders, color variation, rapid growth) or a benign lesion that requires monitoring (e.g., atypical nevus), respond with 'valid'.
                    2. If it is clearly a wound (e.g., a healing injury, scar, scar tissue) or scar unrelated to skin cancer (e.g., acne, eczema, psoriasis, warts), or if the image is unrelated or lacks sufficient detail, respond with 'invalid'.
                    3. If the image is ambiguous (e.g., early-stage lesions, low-resolution images) or does not provide enough information for a confident determination, respond with 'uncertain'.
                    
                    Respond with only 'valid', 'invalid', or 'uncertain'."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this image possibly to be a skin cancer lesion, either benign or malignant?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=9
        )
        
        validation_result = response.choices[0].message.content.strip().lower()
        return validation_result
    
    except Exception as e:
        print(f"Error in image validation: {e}")
        return 'error' 