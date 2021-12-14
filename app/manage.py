from app import app, WEB_API, classifyChatbot, price_predict
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

# Service For Closed Price Predict
@app.route('/ClosedPrice_Predict', methods=['GET'])
def closePricePredict():
    results = price_predict.predict_data()
    return jsonify({'result': float(results[6])}), 200, {"function": "ClosedPrice_Predict"}

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