import os
import requests
import json

backend_SERVERURL = os.getenv('Heroku_backend')
api_inference = os.getenv('APIINFERENCE_BearerToken')

def get_newslist(news_website, limit):
    my_params = json.dumps({'news_website': news_website, 'limit': limit})
    res = requests.get(backend_SERVERURL+'/newslist', params=my_params)
    results = res.json()
    return results

def get_sentimentBertAPI(predict_data):
    payload = json.dumps({
        "inputs": predict_data
    })
    headers = {
        'urlll': 'https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=The+new+trick+cyber-criminals+use+to+cash+out',
        'Authorization': 'Bearer '+api_inference,
        'Content-Type': 'application/json'
    }
    res = requests.request("POST", "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment", headers=headers, data=payload)
    results = res.json()
    return results