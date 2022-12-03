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
from typing import List, Sequence, Dict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from utils import logging

# TODO: tracking loss and metrics for each epoch

class ImageRestorationModel(tf.keras.Model):
    def __init__(
        self,
        restore_model: tf.keras.Model,
        **kwargs
    ):
        super(ImageRestorationModel, self).__init__(**kwargs)
        assert isinstance(restore_model, tf.keras.Model), "restore_model must be a tf.keras.Model"
        self.restore_model = restore_model
    
    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer = None,
        loss: Dict[str, tf.keras.losses.Loss] = None,
        metrics_fn: Dict[str, callable] = None,
        **kwargs
    ):
        super(ImageRestorationModel, self).compile(**kwargs)
        assert isinstance(metrics_fn, dict), f"Metrics_fn args takes dictionary of metrics functions, got {type(metrics_fn)}"
        assert isinstance(loss, dict), f"Loss args takes dictionary of loss functions, got {type(loss)}"
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer), f"Optimizer must be a tf.keras.optimizers.Optimizer, got {type(optimizer)}"
        
        self.restore_model.compile(optimizer, loss, metrics_fn, **kwargs)

        self.optimizer  = optimizer
        self.loss       = loss
        self.metrics_fn = metrics_fn

    @tf.function
    def train_step(self, inputs:tf.Tensor, target:tf.Tensor, **kwargs):
        '''
        Forward pass, calculate loss, calculate gradients, update weights
        args:
            inputs: 4D tensor of shape (batch_size, height, width, channels)
            target: 4D tensor of shape (batch_size, height, width, channels)
        returns:
            dict of loss and metrics_fn
        '''
        assert inputs.shape == target.shape, "Input and target shapes must be same"
        assert self.optimizer is not None, "Optimizer must be defined, use compile() method"
        assert self.loss is not None, "Loss must be defined, use compile() method"
        assert self.metrics_fn is not None, "Metrics_fn must be defined, use compile() method"

        with tf.GradientTape() as tape:
            outputs = self.restore_model(inputs, training=True)
            loss = {}
            for loss_name, loss_fn in self.loss.items():
                loss[loss_name] = float(loss_fn(target, outputs))
            total_loss = tf.math.add_n(list(loss.values()))
        gradients = tape.gradient(total_loss, self.restore_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.restore_model.trainable_variables))
        
        metrics_dict = {}
        for metric_name, metric_fn in self.metrics_fn.items():
            metrics_dict[metric_name] = float(metric_fn(target, outputs))
        return {"total_loss": float(total_loss), **loss,  **metrics_dict}
    
    @tf.function
    def test_step(self, inputs:tf.Tensor, target:tf.Tensor, **kwargs):
        assert inputs.shape == target.shape, "Input and target shapes must be same"

        outputs = self.restore_model(inputs, training=False)
        loss = {}
        total_loss = 0.0
        for _, loss_fn in self.loss.items():
            loss[f'val_{loss_fn.name}'] = float(loss_fn(target, outputs))
            total_loss = tf.math.add_n(list(loss.values()))

        metrics_dict = {}
        for metric_name, metric_fn in self.metrics_fn.items():
            metrics_dict[f'val_{metric_name}'] = float(metric_fn(target, outputs))
        return {"total_loss": float(total_loss), **loss, **metrics_dict}
    
    def summary(self, **kwargs):
        self.restore_model.summary(**kwargs)
    
    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True, save_only_weights = False, **kwargs):
        if save_only_weights:
            self.restore_model.save_weights(f'{filepath}.h5', overwrite, save_format='h5')
        else:
            self.restore_model.save(filepath, overwrite, include_optimizer, save_format, signatures, options, save_traces, **kwargs)