from app import app, WEB_API, classifyChatbot
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

@app.route('/classifyChatbot', methods=['GET', 'POST'])
def Chatbot_classify():
    if request.method == 'POST':
        results = classifyChatbot.predict('今天的新聞')
    elif request.method == 'PUT':
        results = classifyChatbot.train()
    return jsonify({'result': results}), 200, {"function": "classifyChatbot"}

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