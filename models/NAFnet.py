import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from layers import NAFBlock


if __name__ == "__main__":
    x = tf.random.normal([1, 32, 32, 6])
    
    naf = NAFBlock(filters=3, dropout_rate=0.1, kw=3, kh=3)
    
    y = naf(x)
    print(y.shape)