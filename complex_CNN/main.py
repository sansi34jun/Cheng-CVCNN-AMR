import os, random

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import LearningRateScheduler
import mltools, dataset2016
import complex_CNN as mcl

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
    dataset2016.load_data()
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)
print(X_train.shape)
print(Y_train.shape)
print(Y_train)
classes = mods

# Set up some params
nb_epoch = 30  # number of epochs to train on
batch_size = 400  # training batch size

model = mcl.complex_CNN()


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

X_train = complex_data(132000, 128, X_train)
X_val = complex_data(44000, 128, X_val)

X_train = X_train[:, :, np.newaxis]
#print(X_train)
X_val = X_val[:, :, np.newaxis]
filepath = 'weights/weights.h5'
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_val, Y_val),
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patince=5,
                                                          min_lr=0.0000001),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

                    ]
                    )

#  "Accuracy"
plt.plot(range(1, nb_epoch+1), history.history['accuracy'])
print(history.history['accuracy'])
plt.plot(range(1, nb_epoch+1),history.history['val_accuracy'])
print(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('figure/epoch_acc.png')
plt.show()
plt.close()
# "Loss"
plt.plot(range(1, nb_epoch+1),history.history['loss'])
print(history.history['loss'])
plt.plot(range(1, nb_epoch+1),history.history['val_loss'])
print(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('figure/epoch_loss.png')
plt.show()
plt.close()