from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/upload')
def upload():
    return render_template('upload.html')

@main.route('/chatbot')
def chat():
    return render_template('chat.html') 