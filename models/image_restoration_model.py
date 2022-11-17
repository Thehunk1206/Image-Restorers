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
from typing import List, Sequence 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Model


class ImageRestorationModel(Model):
    def __init__(
        self,
        restore_model: Model,
        **kwargs
    ):
        super(ImageRestorationModel, self).__init__(**kwargs)
        self.restore_model = restore_model
        
    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: Sequence[tf.keras.losses.Loss],
        metrics: dict,
        **kwargs
    ):
        super(ImageRestorationModel, self).compile(**kwargs)
        self.optimizer  = optimizer
        self.loss       = loss
        self.metrics    = metrics
    

    @tf.function
    def train_step(self, inputs:tf.tensor, target:tf.tensor, **kwargs):
        assert inputs.shape == target.shape, "Input and target shapes must be same"

        with tf.GradientTape() as tape:
            outputs = self.restore_model(inputs, training=True)
            loss = 0.0
            for loss_fn in self.loss:
                loss += loss_fn(target, outputs)
        gradients = tape.gradient(loss, self.restore_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.restore_model.trainable_variables))
        
        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            metrics[metric_name] = metric_fn(target, outputs)
        return {"loss": loss, **metrics}
    
    @tf.function
    def test_step(self, inputs:tf.tensor, target:tf.tensor, **kwargs):
        assert inputs.shape == target.shape, "Input and target shapes must be same"

        outputs = self.restore_model(inputs, training=False)
        loss = 0.0
        for loss_fn in self.loss:
            loss += loss_fn(target, outputs)
        
        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            metrics[metric_name] = metric_fn(target, outputs)
        return {"loss": loss, **metrics}