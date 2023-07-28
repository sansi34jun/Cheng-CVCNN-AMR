# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from matplotlib import pyplot as plt
import pickle, random, sys,h5py
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.regularizers import *

from keras.models import model_from_json

import mltools,dataset2016
import rmlmodels.ResNet as mcl
import tensorflow as tf

import csv

(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
    dataset2016.load_data()

X_train=np.expand_dims(X_train,axis=3)
X_test=np.expand_dims(X_test,axis=3)
X_val=np.expand_dims(X_val,axis=3)

print(X_train.shape)
# print(X1_train.shape)
classes = mods

# Set up some params
nb_epoch = 25     # number of epochs to train on
batch_size = 400  # training batch size

model=mcl.ResNet()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#plot_model(model, to_file='model_CLDNN.png',show_shapes=True) # print model
model.summary()



filepath = 'weights/weights.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val,Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patince=5,min_lr=0.0000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),
                # keras.callbacks.TensorBoard(histogram_freq=1,write_graph=True,write_images=True)
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



