# lstm_lstm: emotion recognition from speech= lstm, text=lstm
# created for ATSIT paper 2020
# coded by Bagus Tris Atmaja (bagus@ep.its.ac.id)
# changelog:
# 2020/01/28: create names mlp_iemocap_paa

import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import random as rn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from calc_scores import calc_scores

rn.seed(123)
np.random.seed(99)

# load feature and labels
feat = np.load('../data/feat_ws_3.npy')
vad = np.load('../data/y_egemaps.npy')

#feat = feat[:,:-1]

# remove outlier, < 1, > 5
vad = np.where(vad == 5.5, 5.0, vad)
vad = np.where(vad == 0.5, 1.0, vad)

# standardization
scaled_feature = True

# set Dropout
do = 0.3

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
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

# LOSO is from utterance 7869
X_train = feat[:7869]
X_test = feat[7869:]
y_train = vad[:7869]
y_test = vad[7869:]

# batch_size=min(200, n_samples)
# layers (256, 128, 64, 32, 16)
nn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=180, shuffle=True,
    random_state=9, verbose=0, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    n_iter_no_change=10)

nn = nn.fit(X_train, y_train)
y_predict = nn.predict(X_test)


ccc = []
for i in range(0, 3):
    ccc_, _, _ = calc_scores(y_predict[:, i], y_test[:, i])
    ccc.append(ccc_)
    # print("# ", ccc)

## MODIFICATIONS

data = {}

data["First Eval"] = np.mean(ccc)


TEST_DATA = np.load(
    "../data/MELDRaw/MELD_test_data_no_neutral.npy")
TEST_LABEL = np.load(
    "../data/MELDRaw/MELD_labels_no_neutral.npy")
TRAIN_DATA = np.load(
    "../data/MELDRaw/MELD_train_data_no_neutral.npy")
TRAIN_LABEL = np.load(
    "../data/MELDRaw/MELD_labels_no_neutral_train.npy")
print(TRAIN_LABEL)

scaled_feature = True

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(TEST_DATA)
    scaled_feat = scaler.transform(TEST_DATA)
    TEST_DATA = scaled_feat
else:
    TEST_DATA = TEST_DATA

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(TRAIN_DATA)
    scaled_feat = scaler.transform(TRAIN_DATA)
    TRAIN_DATA = scaled_feat
else:
    TRAIN_DATA = TRAIN_DATA

scaled_vad = False

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

y_predict = nn.predict(TEST_DATA)


ccc = []
for i in range(0, 3):
    ccc_, _, _ = calc_scores(y_predict[:, i], TEST_LABEL[:, i])
    ccc.append(ccc_)
    # print("# ", ccc)

data["Second Eval"] = np.mean(ccc)
data["Second Eval whole"] = ccc


nn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True,
    random_state=9, verbose=1, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    n_iter_no_change=250)



X_train = TRAIN_DATA
X_test = TEST_DATA
y_train = TRAIN_LABEL
y_test = TEST_LABEL

nn = nn.fit(X_train, y_train)
y_predict = nn.predict(X_test)


ccc = []
for i in range(0, 3):
    ccc_, _, _ = calc_scores(y_predict[:, i], y_test[:, i])
    ccc.append(ccc_)
    # print("# ", ccc)

data["Third Eval"] = np.mean(ccc)


script_name = os.path.basename(__file__)
with open('JSONs/' + script_name + '_data.json', 'w') as f:
    json.dump(data, f)

# Results:
#  0.3347105262468933
#  0.5823825252355231
#  0.4583157685040692
