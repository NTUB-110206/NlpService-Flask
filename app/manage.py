from app import app, WEB_API
from flask import jsonify, request
from flask_cors import CORS

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, support_credentials=True)


@app.route('/')
def index():
    return jsonify({'data': 'hello'}), 200, {"function": "root"}

@app.route('/sentiment-bert', methods=['POST'])
def sentimentBert():
    results = WEB_API.get_sentimentBertAPI(request.json["data"])
    return jsonify({'result': results[0]}), 200, {"function": "sentiment-bert"}

# # prototype
# @app.route('/sentiment_predict', methods=['POST'])
# def sentimentPredict():
#     results = WEB_API.get_sentimentPredictAPI(request.json["data"])
#     return jsonify({'result': results[0]}), 200, {"function": "sentiment_predict"}

# # prototype
# @app.route('/category_predict', methods=['POST'])
# def categoryPredict():
#     results = WEB_API.get_categoryPredictAPI(request.json["data"])
#     return jsonify({'result': results[0]}), 200, {"function": "category_predict"}