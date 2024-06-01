import os, random

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mltools, dataset2016
import cnn2d as mcl

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
    dataset2016.load_data()
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)


print(X_train.shape)
print(Y_train.shape)
print(Y_train)
# print(X1_train.shape)
classes = mods



# Set up some params
nb_epoch = 100  # number of epochs to train on
batch_size = 36 # training batch size

model = mcl.model2d(input_shape=(128,1), num_classes=11)


def complex_data(n, m, data):
    data = np.array(data)
    # data=data.tolist()
    complex_data2 = []
    for x in range(n):
        complex_data1 = []
        for y in range(m):
            complex_data1.append(complex(data[x][0][y], data[x][1][y]))
        complex_data2.append(complex_data1)
    complex_data2 = np.array(complex_data2)
    return complex_data2

#print(X_train)
X_train = complex_data(132000, 128, X_train)
X_val = complex_data(44000, 128, X_val)

X_train = X_train[:, :, np.newaxis]
#print(X_train)
X_val = X_val[:, :, np.newaxis]

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=0,
                   patience=20)

mc = ModelCheckpoint('model.h5',
                     monitor='val_categorical_accuracy',
                     mode='max',
                     verbose=0,
                     save_best_only=True)

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=nb_epoch,
                    batch_size=batch_size ,
                    verbose=2,
                    callbacks=[es, mc])
#  "Accuracy"
plt.plot(range(1, nb_epoch + 1), history.history['categorical_accuracy'])
print(history.history['categorical_accuracy'])
plt.plot(range(1, nb_epoch + 1), history.history['val_categorical_accuracy'])
print(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('figure/epoch_acc.png')
plt.show()
plt.close()
# "Loss"
plt.plot(range(1, nb_epoch + 1), history.history['loss'])
print(history.history['loss'])
plt.plot(range(1, nb_epoch + 1), history.history['val_loss'])
print(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('figure/epoch_loss.png')
plt.show()
plt.close()
