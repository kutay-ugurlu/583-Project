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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

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
    hidden_layer_sizes=(512, 256, 128, 64, 32, 16), activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=180, shuffle=True,
    random_state=9, verbose=1, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    n_iter_no_change=10)

nn = nn.fit(X_train, y_train)
y_predict = nn.predict(X_test)

ccc = []
for i in range(0, 3):
    ccc_, _, _ = calc_scores(y_predict[:, i], y_test[:, i])
    ccc.append(ccc_)
    # print("# ", ccc)

print(ccc)
print(np.mean(ccc))


data = {}

data["First Eval"] = np.mean(ccc)


val_data_2 = np.load(
    "C:/Users/Kutay/Desktop/deep_mlp_ser/data/MELDRaw/MELD_test_data_no_neutral.npy")
val_label_2 = np.load(
    "C:/Users/Kutay/Desktop/deep_mlp_ser/data/MELDRaw/MELD_labels_no_neutral.npy")
val_label_2 += 0 *  \
    np.random.randn(val_label_2.shape[0], val_label_2.shape[1])

scaled_feature = True

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(val_data_2)
    scaled_feat = scaler.transform(val_data_2)
    val_data_2 = scaled_feat
else:
    val_data_2 = val_data_2

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

y_predict = nn.predict(val_data_2)


ccc = []
for i in range(0, 3):
    ccc_, _, _ = calc_scores(y_predict[:, i], val_label_2[:, i])
    ccc.append(ccc_)
    # print("# ", ccc)

data["Second Eval"] = np.mean(ccc)
data["Second Eval whole"] = ccc


nn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=250, shuffle=True,
    random_state=9, verbose=0, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    n_iter_no_change=100)


X_train, X_test, y_train, y_test = train_test_split(
    val_data_2, (val_label_2), test_size=0.2, random_state=42)


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


# Results:
#  0.3347105262468933
#  0.5823825252355231
#  0.4583157685040692
