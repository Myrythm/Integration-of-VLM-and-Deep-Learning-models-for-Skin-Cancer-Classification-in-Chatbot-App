from flask import Flask
import config
from routes.main_routes import main
from routes.article_routes import articles
from routes.api_routes import api

# Initialize Flask app
app = Flask(__name__)

# Initialize configuration
config.init_app(app)

# Register blueprints
app.register_blueprint(main)
app.register_blueprint(articles)
app.register_blueprint(api)

if __name__ == '__main__':
    app.run(debug=True) 