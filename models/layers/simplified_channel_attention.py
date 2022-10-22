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
from keras.layers import Layer, Conv2D

try:
    from tlc_avgpool import TlcAvgPool2D
except:
    from layers.tlc_avgpool import TlcAvgPool2D

class SimplifiedChannelAttention(Layer):
    def __init__(self, filters: int, kw: int, kh: int) -> None:
        super(SimplifiedChannelAttention, self).__init__()

        self.filters = filters
        self.kw = kw
        self.kh = kh

        self.avg_pool = TlcAvgPool2D([self.kh, self.kw])

        self.conv     = Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='VALID')

    def call(self, inputs: tf.Tensor, **kwargs):
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"

        attention = self.avg_pool(inputs)
        attention = self.conv(attention)

        return attention * inputs
    
    def get_config(self):
        config = super(SimplifiedChannelAttention, self).get_config()
        config.update({'filters': self.filters, 
                        'kw'    : self.kw, 
                        'kh'    : self.kh
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":

    x = tf.random.normal([1, 32, 32, 3])

    sca = SimplifiedChannelAttention(3, 3, 3)

    y = sca(x, training=True)
    print(y.shape)
