#讀取資料套件faw
import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from collections import defaultdict
import nltk

#資料分割套件
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#資料清理及預處理套件
import re
import requests
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder

#Tensorflow套件
from tensorflow import keras
# !pip install -U keras-tuner
# !pip install keras
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from keras.utils import np_utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

#繪圖套件
import matplotlib.pyplot as plt

#評估模型套件
from sklearn.metrics import accuracy_score

#讀取現有Model
from keras.models import load_model

#設定亂數及分割資料集
RANDOM_STATE = 123
TRAIN_SET_RATIO = 1.00
TEST_SET_RATIO = 0.00
VAL_SET_RATIO = 0.00

#文字預處理
def preprocess_text(text):
    #刪除標點符號
    rm_punctuation = lambda x: x.translate(str.maketrans('', '', string.punctuation + "\'\n\r\t"))

    #去除StopWord
    stop_words = set(stopwords.words('english'))
    rm_stopwords = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    text = text.lower()
    text = rm_punctuation(text)
    text = rm_stopwords(text)
    return text

def lemmatize_text(text):
        to_wordnet_tag = defaultdict(lambda : wordnet.NOUN)
        # to_wordnet_tag['N'] = wordnet.NOUN
        to_wordnet_tag['V'] = wordnet.VERB
        to_wordnet_tag['J'] = wordnet.ADJ
        to_wordnet_tag['R'] = wordnet.ADV
        
        lemmatizer = WordNetLemmatizer()
        lemmata = [lemmatizer.lemmatize(word, pos=to_wordnet_tag[tag[0]]) for (word, tag) in text]
        return lemmata
    
def preprocessing_pipeline(document):
    """
    @params
    - document: (n,1) one dimensional shaped series were each entry is a string
    
    @return
    - return document with preprocessed columns
    """
    df = document.to_frame()
    print('Tokenize Text...')
    df['tokens'] = document.apply(word_tokenize)
    
    print('Build Part of Speech Tags...')
    df['pos'] = df.tokens.apply(pos_tag)
    
    print('Build Lemmata...')
    df['lemma'] = df.pos.apply(lemmatize_text)
    
    return df

def get_corpus():
    data = get_newslist('NULL')
    corpus_list = pd.DataFrame.from_dict(data['data']['news'])
    corpus = corpus_list[["news_title", "category"]]
    corpus.columns = ['text', 'category']
    corpus['category'] = ''
    return corpus

def corpus_prepare():
    corpus = get_corpus()

    #類別出現機率計算
    p = corpus.category.value_counts() / corpus.category.shape[0]

    #分類標
    labels = corpus.category.value_counts().index

    #類別對應數字
    label_map = {}
    for i in range(len(labels)):
        label_map[labels[i]] = i

    #打亂類別及資料計算準確度
    test_set_size = int(corpus.shape[0] * TEST_SET_RATIO)
    baseline_acc = []

    #For Loop 交叉驗證資料
    for i in range(10):
        #隨機取得測試資料
        gt = corpus.category.sample(test_set_size).apply(lambda x : label_map[x])
        #隨機預測
        prediction = np.random.choice(list(label_map.values()), test_set_size, p=p)
        baseline_acc += [accuracy_score(gt, prediction)]

    nltk.download('stopwords')


    #類別編碼
    category_encoder = LabelEncoder()
    category_encoder.fit(corpus.category)

    corpus['text_preprocessed'] = corpus.text.apply(preprocess_text)
    corpus['category_enc'] = category_encoder.transform(corpus.category)

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    #文字處理
    X_data = preprocessing_pipeline(corpus.text_preprocessed)
    corpus = corpus.join(X_data, lsuffix="_left", rsuffix="_right")

    #vector representation for lemmata
    keras_tokenizer = Tokenizer()

    X_train = pd.DataFrame()
    X_train['lemma'] = corpus.lemma
    X_train = X_train.to_numpy().reshape(-1,)
    keras_tokenizer.fit_on_texts(X_train)
    X_train_sequences = keras_tokenizer.texts_to_sequences(X_train)

    #padding: how long is the longest sentence?
    #max_sent_length = max([len(doc) for doc in corpus.lemma])
    max_sent_length = 36

    #pad data to be of uniform length
    X_train_padded = pad_sequences(X_train_sequences, max_sent_length, padding = 'post')

    #單獨的token數量
    word2index = keras_tokenizer.word_index

    #Load embeddings
    word2vec = {}

    # choose dimension
    # 50 turned out to provide a better learning curve
    dimension = 50

    with open(os.path.join('drive/MyDrive/二技資管一甲/下學期/news_dataset/glove.6B.50d.txt'.format(dimension)), encoding = "utf-8") as f:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split() #split at space
            word = values[0]
            vec = np.asarray(values[1:], dtype = 'float32') #numpy.asarray()function is used when we want to convert input to an array.
            word2vec[word] = vec

    #創造embedding矩陣
    embedding_matrix = np.zeros((len(word2index)+1, dimension))
    embedding_vec=[]
    for word, i in tqdm(word2index.items()):
        embedding_vec = word2vec.get(word)
        if embedding_vec is not None:
            embedding_matrix[i] = embedding_vec

    model = keras.models.Sequential()
    model = load_model("/content/drive/MyDrive/Colab Notebooks/模型/category.h5")
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return np.argmax(model.predict(X_train_padded), axis=1)

def get_newslist(trend_id):
    res = requests.get(backend_SERVERURL+'/newslist?category='+trend_id)
    results = res.json()
    return results

def category_predict():
    data = get_newslist('NULL')

    predictions = corpus_prepare()

    for i in range(0, len(data['data']['news'])):
      data['data']['news'][i]['category_id'] = str(int(predictions[i]))

    return data['data']['news']

def post_newslist(newslist):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'news': newslist})
    res = requests.put(backend_SERVERURL+'/newslist', headers=headers, data=payload)
    results = res.json()
    return results

post_result = post_newslist(category_predict())
print(post_result)