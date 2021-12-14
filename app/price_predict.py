import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import keras
from datetime import date, timedelta, datetime

# 回看天數
look_back = 14
# 取得閉盤價api
current_price_api = 'https://min-api.cryptocompare.com/data/price'
# 取得歷史交易數據api
hist_data_api = 'https://min-api.cryptocompare.com/data/v2/histo'
# 宣告虛擬貨幣幣別及價值參考
cryptocurrency = 'BTC'
target_currency = 'USD'
# pandas設定
pd.set_option('expand_frame_repr', False)

sc = MinMaxScaler()

# 取得閉盤價
def get_current_data(from_sym='BTC', to_sym='USD', exchange=''):
  url = current_price_api
  parameters = { 'fsym' : from_sym, 'tsyms' : to_sym }
  
  if exchange:
    print('exchange: ', exchange)
    parameters['e'] = exchange
  
  response = requests.get(url, params=parameters)
  data = response.json()

  return data

# 取得歷史交易數據
def get_hist_data(from_sym='BTC', to_sym='USD', timeframe='day', limit=2000, aggregation=1, exchange=''):
  url = hist_data_api
  url += timeframe
  parameters = {'fsym': from_sym,
          'tsym': to_sym,
          'limit': limit,
          'aggregate': aggregation}

  if exchange:
    print('exchange: ', exchange)
    parameters['e'] = exchange

  response = requests.get(url, params=parameters)

  data = response.json()['Data']['Data']

  return data

# 資料正規化
def data_to_dataframe(data):

  df = pd.DataFrame.from_dict(data)
  
  df['time'] = pd.to_datetime(df['time'], unit='s')
  df.set_index('time', inplace=True)

  return df

# 取出幾天前股價來建立成特徵和標籤資料集
def create_dataset(ds, look_back=1):
  X_data, Y_data = [],[]
  for i in range(len(ds)-look_back):
    X_data.append(ds[i:(i+look_back), 0])
    Y_data.append(ds[i+look_back, 0])
    
  return np.array(X_data), np.array(Y_data)

# 預測價格
def predict_data():
  model = keras.models.Sequential()
  model = load_model("data/predict_price.h5")

  data_threeday = get_hist_data(cryptocurrency, target_currency, 'day', look_back)

  df_inputday = data_to_dataframe(data_threeday)

  df_inputday = df_inputday.drop(['volumefrom', 'conversionType', 'conversionSymbol'], axis = 1)

  column_name = ['open', 'high', 'low', 'close', 'volumeto']
  df_inputday = df_inputday.reindex( columns = column_name )

  data = df_inputday[df_inputday['open'] > 0 ].copy()

  df_test = data
  X_test_set = df_test.iloc[:,3:4].values

  X_test = sc.fit_transform(X_test_set)
  X_test_pred = model.predict(X_test)

  # 將預測值轉換回股價
  X_test_pred_price = sc.inverse_transform(X_test_pred)
  return X_test_pred_price

def gen_predict_pic():
  x1 = predict_data()
  x1[look_back - 1] = np.NaN
  x2 = predict_data()
  x2[0:(look_back - 2)] = np.NaN
  plt.plot(x1, color="blue", label="Current")
  plt.plot(x2, color="red", label="Future")
  plt.title("BTC Price Prediction")
  plt.xlabel("Times")
  plt.ylabel("API Time Price")
  plt.legend()
  fig = plt.gcf()
  img_path = './data/'+datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')+'-trend.jpg'
  fig.savefig(img_path, dpi=200)
  return img_path

def InOrDecrease():
  price_list = predict_data()
  if(price_list[look_back - 2] < price_list[look_back - 1]):
    return '上漲'
  else:
    return '下跌'