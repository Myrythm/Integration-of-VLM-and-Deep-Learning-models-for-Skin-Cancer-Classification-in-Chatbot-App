# Integration of GPT-4o Vision and EfficientNetB3 for Skin Cancer Classification in Chatbot App

A web-based application that helps users detect potential skin cancer through image analysis and provides educational information about skin cancer.

## Project Overview
This project is a Flask-based web application that combines advanced AI technologies:
1. GPT-4 Vision to validate whether the user input is indicative of skin cancer or not.
2. If skin cancer is indicated, the EfficientnetB3 Model will process the input for detailed skin cancer detection.
3. The user will then be redirected to the GPT-4o Chatbot page for further consultation on the detected skin cancer 

## Project Structure

```
├── app.py              # Main application entry point
├── config.py           # Application configuration
├── requirements.txt    # Project dependencies
├── routes/            # Route handlers
│   ├── main_routes.py
│   ├── article_routes.py
│   └── api_routes.py
├── templates/         # HTML templates
│   ├── home.html
│   ├── upload.html
│   ├── chat.html
│   └── article/
├── static/           # Static files (CSS, JS, images)
├── model/           # H5 model files
└── utils/           # Utility functions
```

## Application Flow

1. **Home Page**
   - Users land on the home page
   - Access to main features: Image Upload and Chatbot
   - Navigation to articles

2. **Image Analysis Process**
   a. **Initial Upload**
      - User uploads skin image
      - Image format validation
   
   b. **GPT-4 Vision Validation**
      - Skin Cancer Validation
      - Rejection of unsuitable images
      - User feedback for rejected images
   
   c. **EfficienNetB3 Model Analysis**
      - Processing of validated images
      - Detailed skin condition analysis
      - Results generation
   
   d. **Results Display**
      - Results presentation
      - Redirect users to the chatbot page

3. **Chatbot Interface**
   - Interactive chat interface
   - Can memorize detection results and conversations
   - Educational information about skin conditions

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a folder with the name model, then put the keras model (h5) into the folder
6. Run the application:
   ```bash
   python app.py
   ```

## Dependencies

Key dependencies include:
- Flask
- TensorFlow Keras
- Other dependencies listed in requirements.txt

## Results
### Upload Image Page
![Screenshot 2025-03-16 175944](https://github.com/user-attachments/assets/e4736fcf-7396-4b3d-818c-410685a7f001)
![Screenshot 2025-03-26 173644](https://github.com/user-attachments/assets/ba1a47d1-d573-4b60-b5e0-f828562224b3)


### Chatbot Page
![Screenshot 2025-04-09 131200](https://github.com/user-attachments/assets/dfb052ec-cedb-485f-a07f-3e8b7d9ea401)



## Future Improvements

1. Improved model detection accuracy
2. Additional validated skin cancer content
3. Integration with healthcare providers

