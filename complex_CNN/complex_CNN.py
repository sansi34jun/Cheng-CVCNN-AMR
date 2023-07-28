
from keras.layers import Flatten
import tensorflow as tf
import keras.models as models
import cvnn.layers as complex_layers
from keras.layers import Input, Dense

def complex_CNN():
    tf.random.set_seed(1)
    init = 'ComplexGlorotUniform'
    acti = 'crelu'
    model = models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(128, 1)))  # Always use ComplexInput at the start

    model.add(complex_layers.ComplexConv1D(128, 3, activation=acti, padding='same',input_shape=(128, 1), kernel_initializer=init))
    print(model)

    model.add(complex_layers.ComplexConv1D(40, 3, activation=acti, padding='same',kernel_initializer=init))
    model.add(complex_layers.ComplexDropout(0.3))
    print(model)
    model.add(Flatten())
    model.add(complex_layers.ComplexDropout(0.3))
    model.add(Dense(128, activation='relu', name='dense1'))
    model.add(Dense(11, activation='softmax', name='dense2'))  # 修改cart_softmax
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    print(model.summary())
    return model
