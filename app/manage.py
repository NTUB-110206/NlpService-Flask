from app import app, WEB_API, classifyChatbot, sentiment_predict, category_predict
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
        results = classifyChatbot.predict(request.json["msg"])
    elif request.method == 'PUT':
        results = classifyChatbot.train()
    return jsonify({'result': results}), 200, {"function": "classifyChatbot"}

# Service For News Sentiment Predict
@app.route('/sentiment_predict', methods=['GET'])
def sentimentPredict():
    results = sentiment_predict.post_newslist(sentiment_predict.sentiment_predict())
    return jsonify({'result': results[0]}), 200, {"function": "sentiment_predict"}

# # prototype
@app.route('/category_predict', methods=['GET'])
def categoryPredict():
    results = category_predict.post_newslist(category_predict.category_predict())
    return jsonify({'result': results[0]}), 200, {"function": "category_predict"}
