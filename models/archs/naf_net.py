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
import tensorflow.keras as keras
from keras.layers import Conv2D
from keras import Sequential

try:
    from layers import NAFBlock
    from layers import PixelShuffle
except:
    from archs.layers import NAFBlock
    from archs.layers import PixelShuffle

class NAFnet(tf.keras.Model):
    def __init__(
        self,
        width:int  = 16,
        num_middle_blocks: int = 1,
        num_enc_blocks: List[int] = [1, 1, 1, 28],
        num_dec_blocks: List[int] = [1, 1, 1, 1],
        train_size: List = [256, 256, 3],
        dropout_rate: float = 0.0,
        local_agg: bool = False,
        tlc_factor: float = 1.5,
        **kwargs
    ) -> None:
        super(NAFnet, self).__init__(**kwargs)

        self.width              = width
        self.num_middle_blocks  = num_middle_blocks
        self.num_enc_blocks     = num_enc_blocks
        self.num_dec_blocks     = num_dec_blocks
        self.train_size         = train_size
        self.dropout_rate       = dropout_rate
        self.local_agg          = local_agg
        self.tlc_factor         = tlc_factor
        kh, kw                  = int(self.train_size[0] * self.tlc_factor), \
                                    int(self.train_size[1] * self.tlc_factor)
        num_stages              = len(self.num_enc_blocks)

        self.rgb_to_features    = Conv2D(
                                    filters=self.width,
                                    kernel_size = 3,
                                    strides = 1,
                                    padding = 'SAME',
                                    activation=None
                                    )
        self.features_to_rgb    = Conv2D(
                                    filters=3,
                                    kernel_size = 3,
                                    strides = 1,
                                    padding = 'SAME',
                                    activation=None
                                    )
        
        # Encoder and Downsample blocks
        self.encoders_blks     = []
        self.downsample_blks   = []
        for i, num_blocks in enumerate(self.num_enc_blocks):
            # NAF blocks
            self.encoders_blks.append(
                Sequential([
                    NAFBlock(self.width * (2 ** i), self.dropout_rate, kw//(2 ** i), kh//(2 ** i), local_agg=self.local_agg) for _ in range(num_blocks)
                ])
            )
            # Downsample blocks
            self.downsample_blks.append(
                Conv2D(
                    self.width * (2 ** (i + 1)),
                    kernel_size = 2,
                    strides = 2,
                    padding = 'SAME',
                    activation=None
                )
            )
        
        # Middle blocks
        self.middle_blks = Sequential([
            NAFBlock(self.width * (2 ** num_stages), self.dropout_rate, kw//(2 ** num_stages), kh//(2 ** num_stages), local_agg=self.local_agg) for _ in range(self.num_middle_blocks)
        ])

        # Decoder and Upsample blocks
        self.decoders_blks     = []
        self.upsample_blks     = []

        for i, num_blocks in enumerate(self.num_dec_blocks):
            #upsample blocks
            self.upsample_blks.append(
                Sequential([
                    Conv2D(
                        self.width * (2 ** (num_stages - i) * 2), # 2x the number of filters since we are upsampling using pixel shuffle
                        kernel_size=1,
                        strides=1,
                        padding='VALID',
                        activation=None
                    ),
                    PixelShuffle(2)
                ])
            )
            # NAF blocks
            self.decoders_blks.append(
                Sequential([
                    NAFBlock(self.width * (2 ** (num_stages - (i + 1))), self.dropout_rate, kw//(2 ** (num_stages - (i+1))), kh//(2 ** (num_stages - (i+1))), local_agg=self.local_agg) for _ in range(num_blocks)
                ])
            )
    
    def call(self,inputs: tf.Tensor, training=None,**kwargs) -> tf.Tensor:

        if training is None:
            training = False
        # Encoder
        x = self.rgb_to_features(inputs)

        encoder_outputs = [] # store the outputs of the encoder blocks for skip connections
        for encoder, downsample in zip(self.encoders_blks, self.downsample_blks):
            x = encoder(x, training=training)
            encoder_outputs.append(x)
            x = downsample(x, training=training)

        # Middle
        x = self.middle_blks(x, training=training)

        # Decoder
        for (upsample, decoder, encoder_skip) in zip(self.upsample_blks, self.decoders_blks, encoder_outputs[::-1]):
            x = upsample(x, training=training)
            x = x + encoder_skip
            x = decoder(x, training=training)
        
        # Output
        x_res = self.features_to_rgb(x)
        x     = x_res + inputs

        return x
    
    def summary(self, **kwargs):
        x = tf.keras.Input(shape=[None,None,3])
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=True))

        return model.summary(**kwargs)

if __name__ == "__main__":
    from model_profiler import model_profiler

    Batch_size = 2
    use_units = ['GPU IDs', 'MFLOPs', 'GB', 'Million', 'MB']

    # x = tf.random.normal([1, 256, 256, 3])
    x = tf.keras.Input(shape=[None,None,3])
    model = NAFnet(
        width=8,
        num_middle_blocks=1,
        num_enc_blocks=[1, 1, 1, 8],
        num_dec_blocks=[1, 1, 1, 1],
        train_size=[256, 256, 3],
        dropout_rate=0.0,
        local_agg=False,
        tlc_factor=1.5
    )
    model.summary()

    model(x)

    profile = model_profiler(model, Batch_size, use_units=use_units)

    print(profile)