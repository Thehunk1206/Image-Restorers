import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

class SimpleGate(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleGate, self).__init__(**kwargs)
        
    def call(self, inputs: tf.Tensor, **kwargs):
        assert inputs.shape.ndims == 4, "inputs must be 4D tensor"

        x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=-1)

        return x1 * x2
    
    def get_config(self):
        config = super(SimpleGate, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
if __name__ == "__main__":
    x = tf.random.normal([1, 32, 32, 6])
    
    sg = SimpleGate()
    
    y = sg(x)
    print(y.shape)