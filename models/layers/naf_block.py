import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import List, Tuple, Union


import tensorflow as tf
from tensorflow import keras
from keras.layers import LayerNormalization, Conv2D, DepthwiseConv2D, Dense, Dropout


from simple_gate import SimpleGate
from simplified_channel_attention import SimplifiedChannelAttention

class NAFBlock(keras.layers.Layer):
    def __init__(
        self,
        filters              : int,
        dropout_rate         : float,
        kw                   : int,
        kh                   : int,
        depth_wise_expansion : int = 2,
        ffn_expansion        : int = 2,
        **kwargs
    ) -> None:
        super(NAFBlock, self).__init__(**kwargs)

        self.filters              = filters
        self.dropout_rate         = dropout_rate
        self.kw                   = kw
        self.kh                   = kh
        self.depth_wise_filters   = self.filters * depth_wise_expansion
        self.ffn_filters          = self.filters * ffn_expansion

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
                                        SimplifiedChannelAttention(filters= self.filters, kw=self.kw, kh=self.kh),
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

    def call(self, inputs: tf.Tensor, **kwargs):
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"
        
        x = self.spatial_block(inputs)
        x = self.dropout_1(x)
        x = x + inputs

        y = self.channel_block(x)
        y = self.dropout_2(y)
        y = y + x

        return y
    
    def get_config(self):
        config = super(NAFBlock, self).get_config()
        config.update({'filters'             : self.filters, 
                        'dropout_rate'        : self.dropout_rate, 
                        'kw'                  : self.kw, 
                        'kh'                  : self.kh,
                        'depth_wise_expansion': self.depth_wise_expansion,
                        'ffn_expansion'       : self.ffn_expansion
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


