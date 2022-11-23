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

import tensorflow as tf
from tensorflow import keras

class PixelShuffle(keras.layers.Layer):
    def __init__(self, upsample_factor, **kwargs) -> None:
        super(PixelShuffle, self).__init__(**kwargs)

        self.upsample_factor = upsample_factor
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"
        assert inputs.shape[-1] % (self.upsample_factor ** 2) == 0, "input channel must be divisible by upsample_factor ** 2"

        out = tf.nn.depth_to_space(inputs, self.upsample_factor)

        return out

    def get_config(self) -> dict:
        config = super(PixelShuffle, self).get_config()
        config.update({'upsample_factor': self.upsample_factor})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    x = tf.random.normal([1, 32, 32, 512])
    
    ps = PixelShuffle(2)
    
    y = ps(x)
    print(y.shape)
