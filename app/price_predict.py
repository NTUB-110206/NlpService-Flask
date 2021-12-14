import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

look_back = 59
pd.set_option('expand_frame_repr', False)

cryptocurrency = 'BTC'
target_currency = 'USD'

def get_current_data(from_sym='BTC', to_sym='USD', exchange=''):
  url = 'https://min-api.cryptocompare.com/data/price'
  parameters = {'fsym': from_sym, 'tsyms': to_sym}

  if exchange:
    print('exchange: ', exchange)
    parameters['e'] = exchange

  response = requests.get(url, params=parameters)
  data = response.json()

  return data


def get_hist_data(from_sym='BTC', to_sym='USD', timeframe='day', limit=2000, aggregation=1, exchange=''):
  url = 'https://min-api.cryptocompare.com/data/v2/histo'
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


def data_to_dataframe(data):

    df = pd.DataFrame.from_dict(data)

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df

