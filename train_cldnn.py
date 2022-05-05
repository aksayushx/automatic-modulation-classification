
from scipy.io import loadmat
from pandas import factorize
import pickle
import numpy as np
import random
from scipy import signal

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from plot import SNR_accuracy, All_SNR_show_confusion_matrix, plot_split_distribution, SNR_show_confusion_matrix
from process import load_dataset, train_test_valid_split, normalize_data


dataset_pkl = open('./RML2016.10a/RML2016.10a_dict.pkl','rb')
RML_dataset_location = pickle.load(dataset_pkl, encoding='bytes')

SNR, X, modulations, one_hot, lbl_SNR = load_dataset(RML_dataset_location)

mods = []
for i in range(len(modulations)):
    modu = modulations[i].decode('utf-8')
    mods.append(modu)

train_idx, valid_idx, test_idx, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = train_test_valid_split(X, one_hot, train_split=0.7, valid_split=0.15, test_split=0.15)
plot_split_distribution(mods, Y_train, Y_valid, Y_test)

X_train, X_valid, X_test = normalize_data(X_train, X_valid, X_test)


layer_in = keras.layers.Input(shape=(128,2))
layer = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu')(layer_in)
layer = keras.layers.MaxPool1D(pool_size=2)(layer)
layer = keras.layers.LSTM(64, return_sequences=True,)(layer)
layer = keras.layers.Dropout(0.4)(layer)
layer = keras.layers.LSTM(64, return_sequences=True,)(layer)
layer = keras.layers.Dropout(0.4)(layer)
layer = keras.layers.Flatten()(layer)
layer_out = keras.layers.Dense(len(mods), activation='softmax')(layer)

model_cldnn = keras.models.Model(layer_in, layer_out)

optimizer = keras.optimizers.Adam(learning_rate=0.0007)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "cldnn_model.h5", save_best_only=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=5, min_lr=0.000007),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, verbose=1)]

model_cldnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model_cldnn.summary()

tf.keras.backend.clear_session()
history = model_cldnn.fit(X_train, Y_train, batch_size=128, epochs=50, verbose=2, validation_data= (X_valid, Y_valid), callbacks=callbacks)

model = keras.models.load_model("cldnn_model.h5")

test_loss, test_acc = model.evaluate(X_test, Y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

SNR_accuracy(SNR, lbl_SNR, test_idx, X_test, Y_test, model, SNR, 'CNN')


model = tf.keras.models.load_model("./models/cldnn_model.h5")

prediction = model.predict([X_test[:,:,:]])

Y_Pred = []; Y_Test = []; Y_Pred_SNR = []; Y_Test_SNR = []; 
for i in range(len(prediction[:,0])):
    Y_Pred.append(np.argmax(prediction[i,:]))
    Y_Test.append(np.argmax(Y_test[i]))

Y_Pred[:20], Y_Test[:20]


All_SNR_show_confusion_matrix([X_test], model, Y_test, mods, save=False)
SNR_show_confusion_matrix([-6,0,8], lbl_SNR[:], X_test, model, Y_test, test_idx, mods, save=False)