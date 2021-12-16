from app import app, WEB_API, classifyChatbot, sentiment_predict, category_predict, price_predict, WEB_API
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

@app.route('/classifyChatbot', methods=['POST', 'PUT'])
def Chatbot_classify():
    if request.method == 'POST':
        results, prob = classifyChatbot.predict(request.json["msg"])
    elif request.method == 'PUT':
        results = classifyChatbot.train()
    return jsonify({'result': results, 'prob': prob}), 200, {"function": "classifyChatbot"}

# Service For News Sentiment Predict
@app.route('/sentiment_predict', methods=['GET'])
def sentimentPredict():
    results = sentiment_predict.post_newslist(sentiment_predict.sentiment_predict())
    return results
    # return jsonify({'result': results['result']}), 200, {"function": "sentiment_predict"}

# Service For News Category Predict
@app.route('/category_predict', methods=['GET'])
def categoryPredict():
    results = category_predict.post_newslist(category_predict.category_predict())
    return results
    # return jsonify({'result': results['result']}), 200, {"function": "category_predict"}

# Service For Closed Price Predict
@app.route('/ClosedPrice_Predict', methods=['GET'])
def closePricePredict():
    results = price_predict.predict_data()
    return jsonify({'result': float(results[(price_predict.look_back)])}), 200, {"function": "ClosedPrice_Predict"}

# Service For Closed Price Picture Predict
@app.route('/ClosedPricePic_Predict', methods=['GET'])
def closePricePicturePredict():
    results = WEB_API.imgur_upload(price_predict.gen_predict_pic())
    return jsonify({'result': results['link']}), 200, {"function": "ClosedPricePic_Predict"}

# Service For Closed Price Trend Predict
@app.route('/ClosedPriceInOrDecrease_Predict', methods=['GET'])
def closePriceInOrDecreasePredict():
    results = price_predict.InOrDecrease()
    return jsonify({'result': results}), 200, {"function": "ClosedPriceInOrDecrease_Predict"}