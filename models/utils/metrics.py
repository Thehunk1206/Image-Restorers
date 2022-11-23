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
from typing import Dict, List

import tensorflow as tf
from tensorflow import keras

'''
Implement your own metrics here and add them to 
the metrics dictionary 'METRICS_DICT' below.
'''

def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    assert y_true.shape.ndims == 4, f'y_true must be a 4D tensor(batch_size, height, width, channels), got {y_true.shape}'
    assert y_pred.shape.ndims == 4, f'y_pred must be a 4D tensor(batch_size, height, width, channels), got {y_pred.shape}'
    assert y_true.shape == y_pred.shape, f'y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}'

    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    assert y_true.shape.ndims == 4, f'y_true must be a 4D tensor(batch_size, height, width, channels), got {y_true.shape}'
    assert y_pred.shape.ndims == 4, f'y_pred must be a 4D tensor(batch_size, height, width, channels), got {y_pred.shape}'
    assert y_true.shape == y_pred.shape, f'y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}'

    return tf.image.ssim(y_true, y_pred, max_val=1.0)

METRICS_DICT = {
    'psnr': psnr,
    'ssim': ssim
}

def get_metric_fn(metrics: List[str]) -> Dict[str, callable]:
    '''
    Returns a list of metrics functions from the METRICS_DICT
    '''
    assert all([isinstance(metric, str) for metric in metrics]), f'All elements of metrics must be a string, got {[type(metric) for metric in metrics]}'
    
    if isinstance(metrics, str):
        metrics = [metrics]
    
    metrics_dict = {}
    for metric in metrics:
        assert metric in METRICS_DICT.keys(), f'{metric} not found in METRICS_DICT'
        metrics_dict[metric] = METRICS_DICT[metric]
        # metrics_dict[] (METRICS_DICT[metric])
    return metrics_dict

if __name__ == "__main__":
    tf.random.set_seed(23)
    # Create a random image
    img1 = tf.random.uniform((1, 256, 256, 3), minval=0, maxval=1, dtype=tf.float32)
    img2 = tf.random.uniform((1, 256, 256, 3), minval=0, maxval=1, dtype=tf.float32)
    # Compute PSNR
    psnr_val = psnr(img1, img2)
    print(f'PSNR: {psnr_val.numpy()}')

    # Compute SSIM
    ssim_val = ssim(img1, img1)
    print(f'SSIM: {ssim_val.numpy()}')
