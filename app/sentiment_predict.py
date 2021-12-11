import requests, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import re
from keras.models import load_model

backend_SERVERURL = 'https://ntub110206-backend.herokuapp.com'

def data_prepare(data):
    data = data[['text','sentiment']]

    data = data[data.sentiment != "Neutral"]
    data_pos = data[data.sentiment != "Negative"]
    data_neg = data[data.sentiment != "Positive"]
    data_neg = data_neg.sample(n = 2236)
    data_neu = data_pos.sample(n = 2236)

    data = data[data.sentiment != "Negative"]
    data = data.append(data_neg, ignore_index=True)
    data = data.append(data_neu, ignore_index=True)

    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    return data

def corpus_prepare():
    data = pd.read_csv('/data/Sentiment.csv')
    data = data_prepare(data)

    for idx,row in data.iterrows():
        row[0] = row[0].replace('rt',' ')
        
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)

def predict(twt):
    model = Sequential()
    model = load_model("/data/sentiment.h5")
    twt = tokenizer.texts_to_sequences(twt)
    twt = pad_sequences(twt, maxlen = 29, dtype = 'int32', value = 0)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    return np.argmax(sentiment)

def get_newslist(trend_id):
    res = requests.get(backend_SERVERURL+'/newslist?trend='+trend_id)
    results = res.json()
    return results

def sentiment_predict():
    data = get_newslist('NULL')

    corpus_prepare()

    for i in range(0, len(data['data']['news'])):
      data['data']['news'][i]['trend_id'] = str(predict(data['data']['news'][i]['news_title']))
      print(str(predict(data['data']['news'][i]['news_title'])))

    return data['data']['news']

def post_newslist(newslist):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'news': newslist})
    res = requests.put(backend_SERVERURL+'/newslist', headers=headers, data=payload)
    results = res.json()
    return results

post_result = post_newslist(sentiment_predict())
print(post_result)