
from keras.models import Model
from keras.layers import Input, Dense, ReLU, Dropout, Softmax, Conv2D, MaxPool2D,Flatten


def CNN(weights=None,
         input_shape = [2, 128],
         classes=11,
         **kwargs):
    input = Input(input_shape + [1], name='input')

    x = Conv2D(256, (2, 3), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform')(input)
    x = Conv2D(80, (2, 3), padding='same', activation="relu", name="conv3", kernel_initializer='glorot_uniform')(x)
    x=Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='dense1')(x)
    x = Dense(classes, activation='softmax', name='dense2')(x)


    model = Model(inputs=input, outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


import keras
