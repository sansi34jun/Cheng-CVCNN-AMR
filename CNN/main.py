import os, random

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import LearningRateScheduler

import dataset2016
import CNN as mcl

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
    dataset2016.load_data()
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)
print(X_train.shape)
print(Y_train.shape)
print(Y_train)
classes = mods

nb_epoch = 30  # number of epochs to train on
batch_size = 400  # training batch size

model = mcl.CNN()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

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
plt.ylabel('accuracy',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['train', 'validation'], loc='upper left',fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig('figure/epoch_acc.png')
plt.show()
plt.close()
# "Loss"
plt.plot(range(1, nb_epoch+1),history.history['loss'])
print(history.history['loss'])
plt.plot(range(1, nb_epoch+1),history.history['val_loss'])
print(history.history['val_loss'])
plt.title('model loss',fontsize=16)
plt.ylabel('loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['train', 'validation'], loc='upper left',fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig('figure/epoch_loss.png')
plt.show()
plt.close()
