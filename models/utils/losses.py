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
from keras.losses import Loss

class MSE(Loss):
    def __init__(self, weight:float = 1.0, name='mse'):
        super(MSE, self).__init__(name=name)
        assert weight > 0, 'Weight must be greater than 0'

        self.weight = weight

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_true.shape.ndims == 4, f'y_true must be a 4D tensor(batch_size, height, width, channels), got {y_true.shape}'
        assert y_pred.shape.ndims == 4, f'y_pred must be a 4D tensor(batch_size, height, width, channels), got {y_pred.shape}'
        assert y_true.shape == y_pred.shape, f'y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}'

        return self.weight * tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    
    def get_config(self):
        config = super(MSE, self).get_config()
        config.update({'weight': self.weight})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MAE(Loss):
    def __init__(self, weight:float = 1.0, name='mae'):
        super(MAE, self).__init__(name=name)
        assert weight > 0, 'Weight must be greater than 0'
        self.weight = weight

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_true.shape.ndims == 4, f'y_true must be a 4D tensor(batch_size, height, width, channels), got {y_true.shape}'
        assert y_pred.shape.ndims == 4, f'y_pred must be a 4D tensor(batch_size, height, width, channels), got {y_pred.shape}'
        assert y_true.shape == y_pred.shape, f'y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}'

        return self.weight * tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])
        
    def get_config(self):
        config = super(MAE, self).get_config()
        config.update({'weight': self.weight})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CharbonnierLoss(Loss):
    def __init__(self, weight:float = 1.0, name='charbonnier_loss'):
        super(CharbonnierLoss, self).__init__(name=name)
        assert weight > 0, 'Weight must be greater than 0'
        self.weight = weight

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_true.shape.ndims == 4, f'y_true must be a 4D tensor(batch_size, height, width, channels), got {y_true.shape}'
        assert y_pred.shape.ndims == 4, f'y_pred must be a 4D tensor(batch_size, height, width, channels), got {y_pred.shape}'
        assert y_true.shape == y_pred.shape, f'y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}'

        return self.weight * tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)), axis=[1, 2, 3])
        
    def get_config(self):
        config = super(CharbonnierLoss, self).get_config()
        config.update({'weight': self.weight})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PSNRLoss(Loss):
    def __init__(self, max_val:float=1.0, weight:float = 1.0, name:str='psnr_loss'):
        super(PSNRLoss, self).__init__(name=name)
        assert weight > 0, 'Weight must be greater than 0'

        self.max_val    = max_val
        self.weight     = weight
        self.mse        = MSE()
        self.log10      = lambda x: tf.math.log(x) / tf.math.log(10.0)

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_true.shape.ndims == 4, f'y_true must be a 4D tensor(batch_size, height, width, channels), got {y_true.shape}'
        assert y_pred.shape.ndims == 4, f'y_pred must be a 4D tensor(batch_size, height, width, channels), got {y_pred.shape}'
        assert y_true.shape == y_pred.shape, f'y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}'

        return -1.0 * self.weight * 10 * self.log10(self.max_val ** 2 / (self.mse(y_true, y_pred)+1e-10))
    
    def get_config(self):
        config = super(PSNRLoss, self).get_config()
        config.update({'max_val': self.max_val, 'weight': self.weight})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

LOSS_DICT = {
    'mse': MSE,
    'mae': MAE,
    'charbonnier_loss': CharbonnierLoss,
    'psnr_loss': PSNRLoss
}

def get_loss_fn(loss_name:str)->Loss:
    return LOSS_DICT[loss_name]

if __name__ == "__main__":
    tf.random.set_seed(42)
    y_true = tf.random.uniform((1, 128, 128, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    y_pred = tf.random.uniform((1, 128, 128, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

    mae = MAE()
    mse = MSE()
    psnr_scratch = PSNRLoss()
    charbonnier = CharbonnierLoss()

    print(mae(y_true, y_pred))
    print(mse(y_true, y_pred))
    print(psnr_scratch(y_true, y_pred))
    print(charbonnier(y_true, y_pred))