import numpy as np
np.random.seed(1337)

import pandas as pd
from sklearn.externals import joblib
import pickle
from datetime import datetime
import os
import tensorflow as tf
from math import sqrt
import argparse
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO
import h5py
from keras import backend as K
import json

# Keras layers
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Flatten, merge, LSTM, Conv1D, Bidirectional, GRU, Input, concatenate, TimeDistributed
from keras.layers import BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, MaxPooling1D
from keras import utils as np_utils
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras import regularizers
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import TensorBoard

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import jaccard_similarity_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from sklearn.decomposition import PCA

n_in = 24 # Number of time lags
n_out = 48  # Max output hour to be predicted 
out = 20   # Number of output hours to be predicted example out = 4 implies 4h, 8h, 16h, 24h (4 values)

# read the input data
def read_data(gcs_path, columns):
  print('downloading csv file from', gcs_path, columns)
  file_stream = file_io.FileIO(gcs_path, mode='r')
  data = pd.read_csv(
      StringIO(file_stream.read()), header=None,
      names=columns).convert_objects(convert_numeric=True)
  return data


def get_csv(name):
  columns = [
    'PM2.5', 'SO2', 'NO2', 'PM10', 'Ozone', 'CO', 'AT', 'hour', 'month'
  ]
  df = read_data(name, columns)
  df = df.drop(df.index[:2])
  df = df.reset_index(drop=True)
  return df


def series_to_supervised(df, data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  cols, names = list(), list()
  number = list(range(0, n_vars + 1))
  final = dict(zip(number, df.columns.values))
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('%s(t-%d)' % (final[j], i)) for j in final]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('%s(t+%d)' % (final[j], i)) for j in final]
    else:
      names += [('%s(t+%d)' % (final[j], i)) for j in final]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg, names


def encode_df(reframed):
  names = reframed.columns.values
  scaler = MinMaxScaler(feature_range=(0, 1))
  reframed = scaler.fit_transform(reframed)
  reframed = pd.DataFrame(reframed, columns=names)
  return reframed, scaler


def reframe_df(reframed):
  p = list((range(n_in,n_in+n_out)))
  q= [1,4,12,24,48]
  r= [n_in-1]
  nfeats= list(np.asarray(q) + r)
  s,u = list(),list()
  final = [x for x in p if x not in nfeats]
  for i in final:
    t = i *9
    for j in range(9):
      s.append(t+j)
  for i in nfeats:
    t = i *9
    for j in range(4,9):
      s.append(t+j)
  if(n_in>1):
    for i in range(n_in-1):
      t = i *9
      s.append(t+8)
  reframed.drop(reframed.columns[s], axis=1, inplace=True)
  return reframed


def model(train,  job_dir):

  train_X, train_y = train[:, :-out], train[:, -out:]
  train_X = train_X.reshape((train_X.shape[0],1, train_X.shape[1]))
  train_y_list = list()
  for i in range(0, 20, 4):
    train_y_list.append(train_y[:, i])
    i = i + 1
    train_y_list.append(train_y[:, i])
    i = i + 1
    train_y_list.append(train_y[:, i])
    i = i + 1
    train_y_list.append(train_y[:, i])

  inp = Input(shape=(1, 193))
  x = LSTM(
      output_dim=80,
      return_sequences=False,
      kernel_initializer='he_normal',
      kernel_regularizer=regularizers.l2(1e-3),
      activation='relu')(inp)

  x = Dropout(0.2)(x)

  dense_layer_output = []
  class_weight1 = {}
  out_vars = [
      "pm25_1", "so2_1", "no2_1", "pm10_1", "pm25_4", "so2_4", "no2_4",
      "pm10_4", "pm25_12", "so2_12", "no2_12", "pm10_12", "pm25_24", "so2_24",
      "no2_24", "pm10_24", "pm25_48", "so2_48", "no2_48", "pm10_48"
  ]
  train_y_output = {}
  z = 0
  while (z < 20):
    print(out_vars[z])
    train_y_output[out_vars[z]] = train_y_list[z]
    dense_layer_output.append(
        Dense(1, name=out_vars[z])(x))
    z = z + 1
    train_y_output[out_vars[z]] = train_y_list[z]
    dense_layer_output.append(
        Dense(1, name=out_vars[z])(x))
    z = z + 1
    train_y_output[out_vars[z]] = train_y_list[z]
    dense_layer_output.append(
        Dense(1, name=out_vars[z])(x))
    z = z + 1
    train_y_output[out_vars[z]] = train_y_list[z]
    dense_layer_output.append(
        Dense(1, name=out_vars[z])(x))
    z = z + 1
  model = Model(inputs=inp, outputs=dense_layer_output)
  model.compile(
      optimizer='adam', loss='mean_squared_error')

  results = model.fit(
      train_X,
      train_y_output,
      epochs=5,
      batch_size=128,
      verbose=0,
      shuffle=False)
  file_name = "model_regressor" + ".h5"
  model.save(file_name)
  with file_io.FileIO(file_name, mode='rb') as input_f:
    with file_io.FileIO(
        os.path.join(job_dir, file_name), mode='wb+') as output_f:
      output_f.write(input_f.read())

  logs_path = job_dir + "export"
  builder = saved_model_builder.SavedModelBuilder(logs_path)
  print(model.inputs[0])

  output={}
  for i in range(20):
    output[out_vars[i]] = model.outputs[i]

  print(output)
  signature = predict_signature_def(
      inputs={'input': model.inputs[0]},
      outputs=output)

  sess = K.get_session()
  builder.add_meta_graph_and_variables(
      sess=sess,
      tags=[tag_constants.SERVING],
      signature_def_map={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
      })
  builder.save()


def evaluate(test_X, test_y):
  acc1, acc2, acc3 = list(), list(), list()
  predictions_1, predictions_2, predictions_3 = list(), list(), list()
  model1 = "model_regressor" + ".h5"
  loaded_model = load_model(model1)
  loaded_model.load_weights(model1)
  loaded_model.compile(
      loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  test_X_lstm = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  predicted_labels = loaded_model.predict(test_X_lstm)
  out_vars = ["pm25_1","so2_1","no2_1","pm10_1",
                "pm25_4","so2_4","no2_4","pm10_4",
                "pm25_12","so2_12","no2_12","pm10_12",
                "pm25_24","so2_24","no2_24","pm10_24",
                "pm25_48","so2_48","no2_48","pm10_48"]
  for i in range(20):
    pred = np.argmax(predicted_labels[i], axis=1)
    print(out_vars[i], jaccard_similarity_score(pred, test_y[:, i].ravel()))


def get_data(train_file):
  df = get_csv(train_file)
  df = df.drop(df.index[0])
  values = df.values
  reframed, names = series_to_supervised(df, values, n_in, n_out)
  reframed = reframe_df(reframed)
  reframed, scaler = encode_df(
      reframed)
  return reframed


def train_model(job_dir, train_file, **args):
  if 'gs://' in job_dir:
    logs_path = job_dir + '/logs/'
  else:
    logs_path = '.' + '/logs/'
  print('Using logs_path located at {}'.format(logs_path))

  reframed = get_data(train_file)
  test_size = int(reframed.shape[0] * .8)
  test = reframed.values[test_size:, :]
  train = reframed.values[:test_size, :]
  test_X, test_y = test[:, :-out], test[:, -out:]
  model(train, logs_path)
  #evaluate(test_X, test_y)


if __name__ == '__main__':
  # Parse the input arguments for common Cloud ML Engine options
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data')
  parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON')
  args = parser.parse_args()
  arguments = args.__dict__
  train_model(**arguments)
