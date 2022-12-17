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
from  tensorflow import keras as keras
from keras.models import Model
try:
    from naf_net import NAFnet
except:
    from archs.naf_net import NAFnet

_model_names = [
    'nafnet',
]

def get_model(model_name: str, **kwargs) -> Model:
    x = keras.Input(shape=[None, None, 3])

    if model_name not in _model_names:
        raise ValueError(f"Model name should be one of {_model_names}, got {model_name}. \
                            Develop the model and add it to the list at and import the model at {__file__}.")

    if model_name == 'nafnet':
        model = NAFnet(**kwargs)
    
    model   = Model(inputs=x, outputs=model.call(x, training=True))
    return model