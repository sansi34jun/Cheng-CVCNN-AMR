
import tensorflow.keras.models as models
from tensorflow import keras
from tensorflow.keras import layers

import cvnn.layers as complex_layers
from cv2_CNN2.CNBF.ops.complex_layers import *
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten, Lambda, Permute


# %%


def model2d(input_shape, num_classes):
    model = models.Sequential()

    init = 'ComplexGlorotUniform'
    acti = 'zrelu'
    model.add(complex_layers.ComplexInput(input_shape=(128, 1)))  # Always use ComplexInput at the start

    # LFLB1
    model.add(complex_layers.ComplexConv1D(128, 3, input_shape=(128, 1), activation=acti,
                                           kernel_initializer=init))
    model.add(complex_layers.core.ComplexBatchNormalization())
    model.add(layers.Activation(acti))
    model.add(complex_layers.core.ComplexDropout(0.2))
    model.add(complex_layers.pooling.ComplexAvgPooling1D())



    # LFLB2
    model.add(complex_layers.ComplexConv1D(128, 3, activation=acti,  kernel_initializer=init))

    model.add(complex_layers.core.ComplexBatchNormalization())
    model.add(layers.Activation(acti))
    model.add(complex_layers.core.ComplexDropout(0.2))
    model.add(complex_layers.pooling.ComplexAvgPooling1D())
    # LFLB3

    model.add(complex_layers.ComplexConv1D(128, 3, activation=acti,  kernel_initializer=init))
    model.add(complex_layers.core.ComplexBatchNormalization())
    # model.add(layers.MaxPooling2D())#修改参数
    #model.add(complex_layers.core.ComplexBatchNormalization())
    model.add(layers.Activation(acti))
    model.add(complex_layers.core.ComplexDropout(0.3))
    model.add(complex_layers.pooling.ComplexAvgPooling1D())

    # LFLB4

    #model.add(complex_layers.ComplexConv1D(128, 3, activation=acti,  kernel_initializer=init))
    #model.add(complex_layers.core.ComplexBatchNormalization())
    # model.add(layers.MaxPooling2D())#修改参数
    #model.add(complex_layers.core.ComplexBatchNormalization())
    #model.add(layers.Activation(acti))
    #model.add(complex_layers.core.ComplexDropout(0.1))
    #model.add(complex_layers.pooling.ComplexAvgPooling1D())
    print(model.input)
    print(model.output)

    #model.add(layers.Reshape((128, 1, 40)))  # ValueError: not enough values to unpack (expected 4, got 3)

    print(model.input)
    print(model.output)

    model.add(Complex_LSTM(units=77, return_sequences=False))
    # model.add(layers.Reshape([50]))
    #model.add(complex_layers.core.ComplexDropout(0.2))
    print(model.output)
    #model.add(complex_layers.core.ComplexDense(units=num_classes))
    model.add(Dense(units=num_classes,activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0002, decay=0.0,)  # 调整learning_rate,增大

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy']
                  )
    model.summary()
    return model
