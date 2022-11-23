'''
MIT License

Copyright (c) 2022 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow import keras

try:
    from layers.edge_padding2D import EdgePadding2D
except:
    from archs.layers.edge_padding2D import EdgePadding2D

class TlcAvgPool2D(keras.layers.Layer):
    def __init__(
            self,
            kernel_size: Union[List,int, Tuple] = None,
            **kwargs
        ):
        super(TlcAvgPool2D, self).__init__(**kwargs)

        self.kernel_size = kernel_size

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]
    
    def call(self, inputs: tf.Tensor, training:bool):
        '''
        inputs: [batch_size, height, width, channels]
        '''
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"

        if training:
            return tf.reduce_mean(inputs, axis=[1,2], keepdims=True)

        _, h, w, _          = inputs.shape                                                           # Get input h,w
        kernel_h            = tf.reduce_min([self.kernel_size[0], h])                                # Get kernel h
        kernel_w            = tf.reduce_min([self.kernel_size[1], w])                                # Get kernel w

        inputs              = tf.pad(inputs, [[0,0], [1,0], [1,0], [0,0]])                           # Zero Pad input with 1 pixel on each side
        inputs              = tf.cumsum(tf.cumsum(inputs, axis=2), axis=1)                           # Cumulative sum along h,w. Precompute cumsum to implement submatrix sum
        
        # Submatrix sum
        s1  = tf.slice(inputs,
                        [0, 0, 0, 0],
                        [-1, kernel_h, kernel_w, -1]
                    )
        s2  = tf.slice(inputs,
                        [0, 0, (w - kernel_w)+1, 0],
                        [-1, kernel_w, -1, -1]
                    )
        s3  = tf.slice(inputs,
                        [0, (h - kernel_h)+1, 0, 0],
                        [-1, -1, kernel_w, -1]
                    )
        s4  = tf.slice(inputs,
                        [0, (h - kernel_h)+1, (w - kernel_w)+1, 0],
                        [-1, -1, -1, -1]
                    )
        out = (s4 + s1 - s2 - s3) / tf.cast(kernel_h * kernel_w, tf.float32)

        _, out_h, out_w, _  = out.shape                                                              # Get output h,w
        pad_h               = [(h - out_h) // 2, (h - out_h + 1) // 2]                               # Get padding h
        pad_w               = [(w - out_w) // 2, (w - out_w + 1) // 2]                               # Get padding w       
        out                 = EdgePadding2D(pad_h, pad_w)(out)                                       # Pad output to match input size

        return out
    
    def get_config(self):
        config = super(TlcAvgPool2D, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Test
    inputs = tf.constant(np.random.rand(1, 1280, 720, 3), dtype=tf.float32)
    print(inputs.shape)
    out = TlcAvgPool2D(kernel_size=[256,256])(inputs, training=False)
    print(out.shape)

    # Plot
    plt.figure()
    plt.imshow(inputs[0,:,:,0])
    plt.figure()
    plt.imshow(out[0,:,:,0])
    plt.show()