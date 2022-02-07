# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)
# 2012-12-25: modified fot ser_iemocap_loso_hfs.py
#             feature is either std+mean or std+mean+silence (uncomment line 44)

import os
import json
import numpy as np
import pickle
import pandas as pd

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, TimeDistributed, \
    Bidirectional, Flatten, Convolution1D, \
    Embedding, Dropout, Flatten, BatchNormalization, \
    RNN, concatenate, Activation

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.compat.v1.set_random_seed(1234)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# load feature and labels
feat_train = np.load(
    '../data/feat_hfs_gemaps_msp_train.npy')
feat_test = np.load('../data/feat_hfs_gemaps_msp_test.npy')

feat = np.vstack([feat_train, feat_test])
feat = feat.reshape(feat.shape[0], feat.shape[1], 1)

list_path = '../data/improv_data.csv'
list_file = pd.read_csv(list_path, index_col=None)
data = list_file.sort_values(by=['wavfile'])

vad_train = []
vad_test = []

for index, row in data.iterrows():
    #print(row['wavfile'], row['v'], row['a'], row['d'])
    if int(row['wavfile'][18]) in range(1, 6):
        #print("Process vad..", row['wavfile'])
        vad_train.append([row['v'], row['a'], row['d']])
    else:
        #print("Process..", row['wavfile'])
        vad_test.append([row['v'], row['a'], row['d']])

vad = np.vstack([vad_train, vad_test])

scaled_feature = True

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat.reshape(
        feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat.reshape(
        feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaled_feat.reshape(
        feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaler = scaler.fit(vad)
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad)
    vad = scaled_vad
else:
    vad = vad

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics

print("TRAIN DATA SHAPE: ", vad.shape, feat.shape,
      "############################################################3")


def ccc(gold, pred):
    gold = K.squeeze(gold, axis=-1)
    pred = K.squeeze(pred, axis=-1)
    gold_mean = K.mean(gold, axis=-1, keepdims=True)
    pred_mean = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc = K.constant(2.) * covariance / (gold_var + pred_var +
                                         K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):
    # input (num_batches, seq_len, 1)
    ccc_loss = K.constant(1.) - ccc(gold, pred)
    return ccc_loss


# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model(alpha, beta, gamma):
    # speech network
    input_speech = Input(
        shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = Convolution1D(256, 3, activation='relu')(net_speech)
    net_speech = Convolution1D(128, 12, activation='relu')(net_speech)
    net_speech = Convolution1D(64, 12, activation='relu')(net_speech)
    net_speech = Convolution1D(32, 12, activation='relu')(net_speech)
    net_speech = Convolution1D(64, 12, activation='relu')(net_speech)
    model_speech = Flatten()(net_speech)
    #model_speech = Dropout(0.1, seed=None)(net_speech)

    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name)(model_speech)
                      for name in target_names]
    #model_combined = Dense(3, activation='linear')(model_combined)

    model = Model(input_speech, model_combined)
    #model.compile(loss=ccc_loss, optimizer='rmsprop', metrics=[ccc])
    model.compile(loss=ccc_loss,
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='adam', metrics=[ccc])
    return model


# def main(alpha, beta, gamma):
model = api_model(0.1, 0.5, 0.4)

# 8000 first data of session 5 (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
X_train = feat[:len(feat_train)]
Y_train = vad[:len(feat_train)].T

print(Y_train)

Y_train = Y_train.tolist()
hist = model.fit(X_train, Y_train, batch_size=200,  # best:8
                 validation_split=0.2, epochs=180, verbose=2, shuffle=True,
                 callbacks=[earlystop])
metrik = model.evaluate(feat[len(feat_train):],
                        vad[len(feat_train):].T.tolist())
print(metrik)
print("First Eval CCC ave= ", np.mean(metrik[-3:]))

## MODIFICATIONS

data = {}

data["First Eval"] = np.mean(metrik[-3:])


TEST_DATA = np.load(
    "../data/MELDRaw/MELD_test_data_no_neutral.npy")
TEST_LABEL = np.load(
    "../data/MELDRaw/MELD_labels_no_neutral.npy")
TRAIN_DATA = np.load(
    "../data/MELDRaw/MELD_train_data_no_neutral.npy")
TRAIN_LABEL = np.load(
    "../data/MELDRaw/MELD_labels_no_neutral_train.npy")






scaled_feature = True

scaled_feature = True

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(TEST_DATA)
    scaled_feat = scaler.transform(TEST_DATA)
else:
    TEST_DATA = TEST_DATA

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(TRAIN_DATA)
    scaled_feat = scaler.transform(TRAIN_DATA)
else:
    TRAIN_DATA = TRAIN_DATA



TEST_DATA = TEST_DATA.reshape(TEST_DATA.shape[0], TEST_DATA.shape[1], 1)
TRAIN_DATA = TRAIN_DATA.reshape(TRAIN_DATA.shape[0], TRAIN_DATA.shape[1], 1)



scaled_vad = False

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaler = scaler.fit(TEST_LABEL)
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(TEST_LABEL)
    TEST_LABEL = scaled_vad
else:
    TEST_LABEL = TEST_LABEL

TEST_LABEL = np.transpose(TEST_LABEL)
val_list = TEST_LABEL.tolist()

# Test with first model
metrik_val = model.evaluate(TEST_DATA, val_list)
print(metrik_val)
print("Second Eval CCC ave= ", np.mean(metrik_val[-3:]))
data["Second Eval"] = np.mean(metrik_val[-3:])
data["Second Eval whole"] = metrik_val[-3:]


X_train = TRAIN_DATA
X_test = TEST_DATA
y_train = TRAIN_LABEL
y_test = TEST_LABEL


model = api_model(0.1, 0.5, 0.4)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=100,
                          restore_best_weights=True)
hist = model.fit(X_train, np.transpose(y_train).tolist(), batch_size=200,  # best:8
                 validation_split=0.2, epochs=180, verbose=2, shuffle=True,
                 callbacks=[earlystop])
metrik_val = model.evaluate(X_test, val_list)
print(metrik_val)
print("Third Eval CCC ave= ", np.mean(metrik_val[-3:]))
data["Third Eval"] = np.mean(metrik_val[-3:])


script_name = os.path.basename(__file__)
with open('JSONs/' + script_name + '_data.json', 'w') as f:
    json.dump(data, f)
