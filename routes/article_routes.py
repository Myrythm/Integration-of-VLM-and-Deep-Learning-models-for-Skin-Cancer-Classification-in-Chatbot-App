from flask import Blueprint, render_template

articles = Blueprint('articles', __name__)

@articles.route('/Mengenal-Jenis-Jenis-Kanker-Kulit')
def article1():
    return render_template('article/article1.html')

@articles.route('/Tips-Menjaga-Kesehatan-Kulit')
def article2():
    return render_template('article/article2.html')

@articles.route('/Kenali-Ciri-Ciri-Karsioma-Sel-Basal')
def article3():
    return render_template('article/article3.html')

@articles.route('/Perubahan-Bentuk-Pada-Tahi-Lalat')
def article4():
    return render_template('article/article4.html')

@articles.route('/Cara-Melindungi-Diri-dari-Bahaya-Sinar-UV')
def article5():
    return render_template('article/article5.html')

@articles.route('/Tanda-Tanda-Awal-Kanker-Kulit')
def article6():
    return render_template('article/article6.html') 