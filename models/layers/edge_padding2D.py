import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import List, Tuple, Union


import tensorflow as tf
from tensorflow import keras

class EdgePadding2D(keras.layers.Layer):
    def __init__(
            self,
            h_pad: Union[List, Tuple],
            w_pad: Union[List, Tuple],
            input_format: str = 'NHWC',
            **kwargs
        ):
        super(EdgePadding2D, self).__init__(**kwargs)
        assert isinstance(h_pad, (list, tuple)) and isinstance(w_pad, (list, tuple)), "h_pad and w_pad must be list or tuple"
        assert input_format in ['NHWC', 'NCHW'], "input_format must be 'NHWC' or 'NCHW'"
        assert len(h_pad) == 2 and len(w_pad) == 2, "h_pad and w_pad must be of length 2"

        self.h_pad = h_pad
        self.w_pad = w_pad
        self.input_format = input_format
        
    def call(self, inputs: tf.Tensor, **kwargs):
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"

        x = inputs

        if self.input_format == 'NHWC':
            if self.h_pad[0] != 0:
                x_up = tf.gather(x, indices=[0], axis=1)
                x_up = tf.repeat(x_up, self.h_pad[0], axis=1)
                x = tf.concat([x_up, x], axis=1)
            if self.h_pad[1] != 0:
                x_down = tf.gather(tf.reverse(x, axis=[1]), indices=[0], axis=1)
                x_down = tf.repeat(x_down, self.h_pad[1], axis=1)
                x = tf.concat([x, x_down], axis=1)
            if self.w_pad[0] != 0:
                x_left = tf.gather(x, indices=[0], axis=2)
                x_left = tf.repeat(x_left, self.w_pad[0], axis=2)
                x = tf.concat([x_left, x], axis=2)
            if self.w_pad[1] != 0:
                x_right= tf.gather(tf.reverse(x, axis=[2]), indices=[0], axis=2)
                x_right = tf.repeat(x_right, self.w_pad[1], axis=2)
                x = tf.concat([x, x_right], axis=2)
            return x
        else:
            if self.h_pad[0] != 0:
                x_up = tf.gather(x, indices=[0], axis=2)
                x_up = tf.repeat(x_up, self.h_pad[0], axis=2)
                x = tf.concat([x_up, x], axis=2)
            if self.h_pad[1] != 0:
                x_down = tf.gather(tf.reverse(x, axis=[2]), indices=[0], axis=2)
                x_down = tf.repeat(x_down, self.h_pad[1], axis=2)
                x = tf.concat([x, x_down], axis=2)
            if self.w_pad[0] != 0:
                x_left = tf.gather(x, indices=[0], axis=3)
                x_left = tf.repeat(x_left, self.w_pad[0], axis=3)
                x = tf.concat([x_left, x], axis=3)
            if self.w_pad[1] != 0:
                x_right= tf.gather(tf.reverse(x, axis=[3]), indices=[0], axis=3)
                x_right = tf.repeat(x_right, self.w_pad[1], axis=3)
                x = tf.concat([x, x_right], axis=3)
            return x

    def get_config(self):
        return {
            'h_pad': self.h_pad,
            'w_pad': self.w_pad,
            'input_format': self.input_format
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = tf.random.uniform([1,5,5,3], minval=0, maxval=1)

    layer = EdgePadding2D(h_pad=(2, 1), w_pad=(2, 1), input_format='NHWC')
    y = layer(x)

    print(y.shape)

    if layer.input_format == 'NHWC':
        plt.imshow(tf.squeeze(y))
    else:
        plt.imshow(tf.squeeze(tf.transpose(y, perm=[0, 2, 3, 1])))
    plt.show()