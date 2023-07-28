import tensorflow as tf
from packaging import version
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import conv_utils

if version.parse(tf.__version__) < version.parse("2.6.0"):
    from tensorflow.python.keras.engine.input_spec import InputSpec
else:
    from tensorflow.keras.layers import InputSpec
from tensorflow.python.framework import tensor_shape
from abc import abstractmethod
# Typing
from typing import Union, Optional, Tuple
# Own models
from cvnn.layers.core import ComplexLayer
from cvnn.layers.core import DEFAULT_COMPLEX_TYPE


class ComplexPooling2D(Layer, ComplexLayer):
    """
    Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
    Abstract class. This class only exists for code reuse. It will never be an exposed API.
    """

    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: str = 'valid', data_format: Optional[str] = None,
                 name: Optional[str] = None, dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        """
        :param pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
            specifying the size of the pooling window.
            Can be a single integer to specify the same value for all spatial dimensions.
        :param strides: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for all spatial dimensions.
        :param padding: A string. The padding method, either 'valid' or 'same'. Case-insensitive.
        :param data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, height, width)`.
        :param name: A string, the name of the layer.
        """
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2,
                                                    'pool_size')  # Values are checked here. No need to check them later
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    @abstractmethod
    def pool_function(self, inputs, ksize, strides, padding, data_format):
        pass

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4))
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = super(ComplexPooling2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format,
            'dtype': self.my_dtype
        })
        return config


class ComplexMaxPooling2D(ComplexPooling2D):
    """
    Max pooling operation for 2D spatial data.
    Works for complex dtype using the absolute value to get the max.
    """

    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: str = 'valid', data_format: Optional[str] = None,
                 name: Optional[str] = None, **kwargs):
        super(ComplexMaxPooling2D, self).__init__(pool_size=pool_size, strides=strides, padding=padding,
                                                  data_format=data_format, name=name, **kwargs)
        self.argmax = None

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        # The max is calculated with the absolute value. This will still work on real values.
        if inputs.dtype.is_complex:
            abs_in = tf.math.abs(inputs)
        else:
            abs_in = inputs
        output, argmax = tf.nn.max_pool_with_argmax(input=abs_in, ksize=ksize, strides=strides,
                                                    padding=padding, data_format=data_format,
                                                    include_batch_in_index=True)
        self.argmax = argmax
        shape = tf.shape(output)
        tf_res = tf.reshape(tf.gather(tf.reshape(inputs, [-1]), argmax), shape)
        # assert np.all(tf_res == output)             # For debugging when the input is real only!
        assert tf_res.dtype == inputs.dtype
        return tf_res

    def get_real_equivalent(self):
        return ComplexMaxPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")

    def get_max_index(self):
        if self.argmax is None:
            raise AttributeError("Variable argmax did not exist, call at least once the max-pooling layer")
        return self.argmax  # TODO: Shall I check this is use only once?


class ComplexMaxPooling2DWithArgmax(ComplexMaxPooling2D):
    """
    Max pooling operation for 2D spatial data and outputs both max values and indices.
    This class is equivalent to ComplexMaxPooling2D but that also outputs indices.
    Useful to perform Max Unpooling using ComplexUnPooling2D.
    Works for complex dtype using the absolute value to get the max.
    """

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        """
        :param inputs: A Tensor. Input to pool over.
        :param ksize: An int or list of ints that has length 1, 2 or 4.
            The size of the window for each dimension of the input tensor.
        :param strides: An int or list of ints that has length 1, 2 or 4.
            The stride of the sliding window for each dimension of the input tensor.
        :param padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        :param data_format: An optional string, must be set to "NHWC". Defaults to "NHWC".
            Specify the data format of the input and output data.
        :return: A tuple of Tensor objects (output, argmax).
            - output	A Tensor. Has the same type as input.
            - argmax	A Tensor. The indices in argmax are flattened (Complains directly to TensorFlow)
        """
        # The max is calculated with the absolute value. This will still work on real values.
        if inputs.dtype.is_complex:
            abs_in = tf.math.abs(inputs)
        else:
            abs_in = inputs
        output, argmax = tf.nn.max_pool_with_argmax(input=abs_in, ksize=ksize, strides=strides,
                                                    padding=padding, data_format=data_format,
                                                    include_batch_in_index=True)
        shape = tf.shape(output)
        tf_res = tf.reshape(tf.gather(tf.reshape(inputs, [-1]), argmax), shape)
        # assert np.all(tf_res == output)             # For debugging when the input is real only!
        assert tf_res.dtype == inputs.dtype
        return tf_res, argmax


class ComplexAvgPooling2D(ComplexPooling2D):

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        inputs_r = tf.math.real(inputs)
        inputs_i = tf.math.imag(inputs)
        output_r = tf.nn.avg_pool2d(input=inputs_r, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        output_i = tf.nn.avg_pool2d(input=inputs_i, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        if inputs.dtype.is_complex:
            output = tf.complex(output_r, output_i)
        else:
            output = output_r
        return output

    def get_real_equivalent(self):
        return ComplexAvgPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")

    
class ComplexCircularAvgPooling2D(ComplexPooling2D):

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        inputs_abs   = tf.math.abs(inputs)
        inputs_phase = tf.math.phase(inputs)
        amp = tf.nn.avg_pool2d(input=inputs_abs, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        raise NotImplemetedError("Still not implemented")       # https://en.wikipedia.org/wiki/Circular_mean
        pha = tf.nn.avg_pool2d(input=inputs_phase, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)       # TODO
        output =  tf.cast(tf.complex(amp * tf.math.cos(pha), amp * tf.math.sin(pha)), dtype=z.dtype)
        return output

    def get_real_equivalent(self):
        return ComplexAvgPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")


class ComplexPolarAvgPooling2D(ComplexPooling2D):

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        inputs_abs = tf.math.abs(inputs)
        output_abs = tf.nn.avg_pool2d(input=inputs_abs, ksize=ksize, strides=strides,
                                      padding=padding, data_format=data_format)
        # Use circular mean
        inputs_angle = tf.math.angle(inputs)
        unit_x = tf.math.cos(inputs_angle)      # Convert all angles to corresponding points on the unit circle
        unit_y = tf.math.sin(inputs_angle)      # convert polar coordinates to Cartesian coordinates.
        avg_unit_x = tf.nn.avg_pool2d(input=unit_x, ksize=ksize, strides=strides,   # Then compute the arithmetic
                                      padding=padding, data_format=data_format)     # mean of these points.
        avg_unit_y = tf.nn.avg_pool2d(input=unit_y, ksize=ksize, strides=strides,
                                      padding=padding, data_format=data_format)
        # The angle is a reasonable mean of the input angles.
        output_angle = tf.math.angle(tf.complex(avg_unit_x, avg_unit_y))
        # Unknown result. If the angles are uniformly distributed on the circle,
        # then the resulting radius will be 0, and there is no circular mean.
        if inputs.dtype.is_complex:
            output = tf.complex(output_abs * tf.math.cos(output_angle), output_abs * tf.math.sin(output_angle))
        else:
            output = output_abs
        return output

    def get_real_equivalent(self):
        return ComplexPolarAvgPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                        data_format=self.data_format, name=self.name + "_real_equiv")


class ComplexUnPooling2D(Layer, ComplexLayer):
    """
    Performs UnPooling as explained in:
    https://www.oreilly.com/library/view/hands-on-convolutional-neural/9781789130331/6476c4d5-19f2-455f-8590-c6f99504b7a5.xhtml
    This class was inspired to recreate the CV-FCN model of https://www.mdpi.com/2072-4292/11/22/2653
    As far as I am concerned this class should work for any dimensional input but I have not tested it
        (and you need the argmax which I only implemented the 2D case).
    """

    def __init__(self, desired_output_shape=None, upsampling_factor: Optional[int] = None, name=None,
                 dtype=DEFAULT_COMPLEX_TYPE, dynamic=False, **kwargs):
        """
        :param desired_output_shape: tf.TensorShape (or equivalent like tuple or list).
            The expected output shape without the batch size.
            Meaning that for a 2D image to be enlarged, this is size 3 of the form HxWxC or CxHxW
        :param upsampling_factor: Integer. The factor to which enlarge the image, 
            For example, if upsampling_factor=2, an input image of size 32x32 will be 64x64.
            This parameter is ignored if desired_output_shape is used or if the output shape is given to the call funcion.
        """
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        if desired_output_shape is not None:
            if not tf.TensorShape(desired_output_shape).is_fully_defined():
                # tf.print(f"Warning: Partially defined desired_output_shape will be casted to None")
                # desired_output_shape = None
                raise ValueError(f"desired_output_shape must be fully defined, got {desired_output_shape}")
            elif len(desired_output_shape) != 3:
                raise ValueError(f"desired_output_shape expected to be size 3 and got size {len(desired_output_shape)}")
        self.desired_output_shape = desired_output_shape
        if upsampling_factor is None or isinstance(upsampling_factor, int):
            self.upsampling_factor = upsampling_factor
        else:
            raise ValueError(f"Unsuported upsampling_factor = {upsampling_factor}")
        super(ComplexUnPooling2D, self).__init__(trainable=False, name=name, dtype=self.my_dtype.real_dtype,
                                                 dynamic=dynamic, **kwargs)

    def call(self, inputs, **kwargs):
        """
        TODO: Still has a bug, if argmax has coincident indexes. Don't think this is desired (but might).
        :param inputs: A tuple of Tensor objects (input, argmax).
            - input 	A Tensor.
            - argmax	A Tensor. The indices in argmax are flattened (Complains directly to TensorFlow)
            - output_shape (Optional) A tf.TensorShape (or equivalent like tuple or list).
                The expected output shape without the batch size.
                Meaning that for a 2D image to be enlarged, this is size 3 of the form HxWxC or CxHxW
            # TODO: I could make an automatic unpool mat if it is not given.
        """
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called on a list of inputs.')
        if len(inputs) == 2:
            inputs_values, unpool_mat = inputs
            output_shape = self.desired_output_shape
        elif len(inputs) == 3:
            inputs_values, unpool_mat, output_shape = inputs
        else:
            raise ValueError(f'inputs = {inputs} must have size 2 or 3 and had size {len(inputs)}')

        # https://stackoverflow.com/a/42549265/5931672
        # https://github.com/tensorflow/addons/issues/632#issuecomment-482580850
        # This is for the case I don't know the expected output shape so I used the upsampling factor
        if not tf.TensorShape(output_shape).is_fully_defined():
            if self.upsampling_factor is None:
                raise ValueError('output_shape should be passed as 3rd element or either desired_output_shape '
                                 'or upsampling_factor should be passed on construction')
            if inputs_values.get_shape()[1:].is_fully_defined():
                output_shape = tf.tile(inputs_values,
                                       [1, self.upsampling_factor, self.upsampling_factor, 1]).get_shape()[1:]
            else:
                output_shape = tf.shape(inputs_values)[1:]
        elif self.upsampling_factor is not None:
            tf.print("WARNING: Ignoring self.upsampling_factor parameter")

        flat_output_shape = tf.reduce_prod(output_shape)
        shape = (tf.shape(inputs_values)[0] * flat_output_shape,)
        updates = tf.reshape(inputs_values, [-1])
        indices = tf.expand_dims(tf.reshape(unpool_mat, [-1]), axis=-1)
        # assert indices.shape[-1] == tf.rank(shape)
        ret = tf.scatter_nd(indices, updates, shape=shape)
        # import pdb; pdb.set_trace()
        desired_output_shape_with_batch = tf.concat([[tf.shape(inputs_values)[0]], output_shape], axis=0)
        ret = tf.reshape(ret, shape=desired_output_shape_with_batch)
        return ret

    def get_real_equivalent(self):
        return ComplexUnPooling2D(desired_output_shape=self.desired_output_shape, name=self.name,
                                  dtype=self.my_dtype.real_dtype, dynamic=self.dtype)

    def get_config(self):
        config = super(ComplexUnPooling2D, self).get_config()
        config.update({
            'desired_output_shape': self.desired_output_shape,
            'name': self.name,
            'dtype': self.my_dtype,
            'dynamic': False,
        })
        return config


"""
    3D Pooling
"""


class ComplexPooling3D(Layer, ComplexLayer):
    def __init__(self, pool_size=(2, 2, 1), strides=None,
                 padding='valid', data_format='channels_last',
                 name=None, dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        self.my_dtype = dtype
        super(ComplexPooling3D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    @abstractmethod
    def pool_function(self, inputs, ksize, strides, padding, data_format):
        pass

    def call(self, inputs, **kwargs):
        outputs = self.pool_function(
            inputs,
            self.pool_size,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 5))
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            deps = input_shape[-3]
            rows = input_shape[-2]
            cols = input_shape[-1]
        else:
            deps = input_shape[-4]
            rows = input_shape[-3]
            cols = input_shape[-2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], deps, rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], deps, rows, cols, input_shape[-1]])

    def get_config(self):
        config = super(ComplexPooling3D, self).get_config()
        config.update({
            'strides': self.strides,
            'pool_size': self.pool_size,
            'padding': self.padding,
            'data_format': self.data_format,
            'dtype': self.my_dtype
        })
        return config


class ComplexAvgPooling3D(ComplexPooling3D):

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        inputs_r = tf.math.real(inputs)
        inputs_i = tf.math.imag(inputs)
        output_r = tf.nn.avg_pool3d(input=inputs_r, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        output_i = tf.nn.avg_pool3d(input=inputs_i, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        if inputs.dtype.is_complex:
            output = tf.complex(output_r, output_i)
        else:
            output = output_r
        return output

    def get_real_equivalent(self):
        return ComplexAvgPooling3D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")


"""
    1D Pooling
"""


class ComplexPooling1D(Layer, ComplexLayer):
    def __init__(self, pool_size=2, strides=None,
                 padding='valid', data_format='channels_last',
                 name=None, dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        self.my_dtype = dtype
        super(ComplexPooling1D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=3)

    @abstractmethod
    def pool_function(self, inputs, ksize, strides, padding, data_format):
        pass

    def call(self, inputs, **kwargs):
        outputs = self.pool_function(
            inputs,
            self.pool_size,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 3))
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size[0],
                                               self.padding,
                                               self.strides[0])
        if self.data_format == 'channels_first':
            return tf.TensorShape([input_shape[0], features, length])
        else:
            return tf.TensorShape([input_shape[0], length, features])

    def get_config(self):
        config = super(ComplexPooling1D, self).get_config()
        config.update({
            'strides': self.strides,
            'pool_size': self.pool_size,
            'padding': self.padding,
            'data_format': self.data_format,
            'dtype': self.my_dtype
        })
        return config


class ComplexAvgPooling1D(ComplexPooling1D):

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        inputs_r = tf.math.real(inputs)
        inputs_i = tf.math.imag(inputs)
        output_r = tf.nn.avg_pool1d(input=inputs_r, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        output_i = tf.nn.avg_pool1d(input=inputs_i, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        if inputs.dtype.is_complex:
            output = tf.complex(output_r, output_i)
        else:
            output = output_r
        return output

    def get_real_equivalent(self):
        return ComplexAvgPooling1D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")
