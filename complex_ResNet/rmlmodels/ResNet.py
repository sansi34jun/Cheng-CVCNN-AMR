import os

from keras.models import Model
from keras.layers import Input,Dense,ReLU,Dropout,Activation,concatenate,Softmax,Conv2D,MaxPool2D,Add,BatchNormalization
from keras.layers import Bidirectional,Flatten,CuDNNGRU
from keras.utils.vis_utils import plot_model
import cvnn.layers as complex_layers
import keras.models as models
import tensorflow as tf
def ResNet(weights=None,
           input_shape=None,
             classes=11,
             **kwargs):
    if input_shape is None:
        input_shape = [128]
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr=0.6
    input1 = Input(input_shape+[1],name='input')

    model1 = models.Sequential()
    model = models.Sequential()

    init = 'ComplexGlorotUniform'
    acti = 'cart_relu'
    model.add(complex_layers.ComplexInput(input_shape=(128, 1)))

    model.add(complex_layers.ComplexConv1D(64, 3, activation=acti, input_shape=(128, 1), kernel_initializer=init, padding='same'))
    print(model)
    # x = Dropout(dr)(x)
    model.add(complex_layers.ComplexConv1D(64, 3, activation=acti, input_shape=(128, 1), kernel_initializer=init, padding='same'))
    # x = Dropout(dr)(x)
    print(model)
    print(model1)
    model.add(model1)

    #x1 = Activation('relu')(x1)
    model.add(complex_layers.ComplexConv1D(40, 3, activation=acti, input_shape=(128, 1), kernel_initializer=init, padding='same'))
    model.add(complex_layers.ComplexConv1D(40, 3, activation=acti, input_shape=(128, 1), kernel_initializer=init, padding='same'))
    #x = Dropout(dr)(x)
    model.add(complex_layers.ComplexFlatten())
    model.add(Dense(128, activation='relu', name='fc1'))

    model.add(Dense(classes, activation='softmax', name='softmax'))





    return model

