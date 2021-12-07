from app import app
from flask import jsonify
from flask_cors import CORS

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, support_credentials=True)


@app.route('/')
def index():
    return jsonify({'data': 'hello'}), 200, {"function": "root"}
