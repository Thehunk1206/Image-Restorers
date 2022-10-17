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
        # assert inputs.shape[-1] % (self.upsample_factor ** 2) == 0, "input channel must be divisible by upsample_factor ** 2"

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
    x = tf.random.normal([1, 32, 32, 12])
    
    ps = PixelShuffle(2)
    
    y = ps(x)
    print(y.shape)
