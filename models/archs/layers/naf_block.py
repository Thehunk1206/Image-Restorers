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
from keras.layers import Layer, LayerNormalization, Conv2D, DepthwiseConv2D, Dense, Dropout

try:
    from layers.simple_gate import SimpleGate
    from layers.simplified_channel_attention import SimplifiedChannelAttention
except:
    from archs.layers.simple_gate import SimpleGate
    from archs.layers.simplified_channel_attention import SimplifiedChannelAttention

class NAFBlock(Layer):
    def __init__(
        self,
        filters              : int,
        dropout_rate         : float,
        kw                   : int,
        kh                   : int,
        depth_wise_expansion : int = 2,
        ffn_expansion        : int = 2,
        local_agg            : bool = False,
        **kwargs
    ) -> None:
        super(NAFBlock, self).__init__(**kwargs)

        self.filters              = filters
        self.dropout_rate         = dropout_rate
        self.kw                   = kw
        self.kh                   = kh
        self.depth_wise_expansion = depth_wise_expansion
        self.ffn_expansion        = ffn_expansion
        self.local_agg            = local_agg
        self.depth_wise_filters   = self.filters * self.depth_wise_expansion
        self.ffn_filters          = self.filters * self.ffn_expansion

        self.spatial_block        = keras.Sequential([
                                        LayerNormalization(),
                                        Conv2D(
                                            filters=self.depth_wise_filters, 
                                            kernel_size=1, 
                                            strides=1,
                                            padding='VALID',
                                            activation=None
                                        ),
                                        DepthwiseConv2D(
                                            kernel_size=3,
                                            strides=1,
                                            padding='SAME',
                                            activation=None
                                        ),
                                        SimpleGate(),
                                        SimplifiedChannelAttention(filters= self.filters, kw=self.kw, kh=self.kh, local_agg=self.local_agg),
                                        Conv2D(
                                            filters=self.filters,
                                            kernel_size=1,
                                            strides=1,
                                            padding='VALID',
                                            activation=None
                                        )
                                    ])

        self.dropout_1           = Dropout(self.dropout_rate)

        self.channel_block       = keras.Sequential([
                                        LayerNormalization(),
                                        Conv2D(
                                            filters=self.ffn_filters,
                                            kernel_size=1,
                                            strides=1,
                                            padding='VALID',
                                        ),
                                        SimpleGate(),
                                        Conv2D(
                                            filters=self.filters,
                                            kernel_size=1,
                                            strides=1,
                                            padding='VALID',
                                        )
        ])

        self.dropout_2           = Dropout(self.dropout_rate)

        self.beta                = self.add_weight(
                                        name='beta',
                                        shape=(1, 1, 1, self.filters),
                                        initializer='zeros',
                                        trainable=True,
                                        dtype=tf.float32
                                    )
        self.gamma               = self.add_weight(
                                        name='gamma',
                                        shape=(1, 1, 1, self.filters),
                                        initializer='zeros',
                                        trainable=True,
                                        dtype=tf.float32
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"
        
        x = self.dropout_1(self.spatial_block(inputs)) * self.beta + inputs
        x = self.dropout_2(self.channel_block(x)) * self.gamma + x

        return x
    
    def get_config(self):
        config = super(NAFBlock, self).get_config()
        config.update({'filters'             : self.filters, 
                        'dropout_rate'        : self.dropout_rate, 
                        'kw'                  : self.kw, 
                        'kh'                  : self.kh,
                        'depth_wise_expansion': self.depth_wise_expansion,
                        'ffn_expansion'       : self.ffn_expansion,
                        'local_agg'           : self.local_agg
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
if __name__ == "__main__":
    x = tf.random.normal([1, 32, 32, 3])

    naf = NAFBlock(3, 0.1, 4, 4)

    y = naf(x)
    print(y.shape)