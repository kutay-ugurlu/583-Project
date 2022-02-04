# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)
# 2012-12-25: modified fot ser_iemocap_loso_hfs.py
#             feature is either std+mean or std+mean+silence (uncomment line 44)

import os
import json
from sklearn.model_selection import train_test_split
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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.compat.v1.set_random_seed(1234)

# load feature and labels
feat = np.load('../data/feat_ws_3.npy')
vad = np.load('../data/y_egemaps.npy')

# for CNN input shape (batch, channel, steps)
feat = feat.reshape(feat.shape[0], feat.shape[1], 1)

# remove outlier, < 1, > 5
vad = np.where(vad == 5.5, 5.0, vad)
vad = np.where(vad == 0.5, 1.0, vad)

# standardization
scaled_feature = False

# set Dropout
do = 0.3

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

# better scaled
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
model.summary()

# 7869 first data of session 5 (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit(feat[:7869], vad[:7869].T.tolist(), batch_size=200,  # best:8
                 validation_split=0.2, epochs=180, verbose=1, shuffle=True,
                 callbacks=[earlystop])
metrik = model.evaluate(feat[7869:], vad[7869:].T.tolist())
print(metrik)
print("CCC ave= ", np.mean(metrik[-3:]))

data = {}

data["First Eval"] = np.mean(metrik[-3:])


val_data_2 = np.load(
    "C:/Users/Kutay/Desktop/deep_mlp_ser/data/MELDRaw/MELD_test_data_no_neutral.npy")
val_data_2 = val_data_2.reshape(val_data_2.shape[0], val_data_2.shape[1], 1)
val_label_2 = np.load(
    "C:/Users/Kutay/Desktop/deep_mlp_ser/data/MELDRaw/MELD_labels_no_neutral.npy")
val_label_2 += .01 * \
    np.random.randn(val_label_2.shape[0], val_label_2.shape[1])

scaled_feature = True

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(val_data_2.reshape(
        val_data_2.shape[0]*val_data_2.shape[1], val_data_2.shape[2]))
    scaled_feat = scaler.transform(val_data_2.reshape(
        val_data_2.shape[0]*val_data_2.shape[1], val_data_2.shape[2]))
    scaled_feat = scaled_feat.reshape(
        val_data_2.shape[0], val_data_2.shape[1], val_data_2.shape[2])
    val_data_2 = scaled_feat
else:
    val_data_2 = val_data_2

scaled_vad = False

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaler = scaler.fit(val_label_2)
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(val_label_2)
    val_label_2 = scaled_vad
else:
    val_label_2 = val_label_2

val_label_2 = np.transpose(val_label_2)
val_list = val_label_2.tolist()

# Test with first model
metrik_val = model.evaluate(val_data_2, val_list)
print(metrik_val)
print("Second Eval CCC ave= ", np.mean(metrik_val[-3:]))
data["Second Eval"] = np.mean(metrik_val[-3:])
data["Second Eval whole"] = metrik_val[-3:]

# Train Test Split
print(val_data_2.shape, val_label_2.shape)
print(val_label_2)
X_train, X_test, y_train, y_test = train_test_split(
    val_data_2, np.transpose(val_label_2), test_size=0.33, random_state=42)


model = api_model(0.1, 0.5, 0.4)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=100,
                          restore_best_weights=True)
hist = model.fit(X_train, np.transpose(y_train).tolist(), batch_size=200,  # best:8
                 validation_split=0.2, epochs=180, verbose=2, shuffle=True,
                 callbacks=[earlystop])
metrik_val = model.evaluate(X_test, np.transpose(y_test).tolist())
print(metrik_val)
print("Third Eval CCC ave= ", np.mean(metrik_val[-3:]))
data["Third Eval"] = np.mean(metrik_val[-3:])


script_name = os.path.basename(__file__)
with open('JSONs/' + script_name + '_data.json', 'w') as f:
    json.dump(data, f)
