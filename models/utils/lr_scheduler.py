import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 


'''
If you want to use a custom learning rate scheduler, 
you can do so by creating a class that inherits from `tf.keras.optimizers.schedules.LearningRateSchedule` and 
implementing the __call__ method. After implementing the class, add it to the `SCHEDULERS_DICT` dictionary.
'''

SCHEDULER_DICT = {
    'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay,
    'PiecewiseConstantDecay': tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    'PolynomialDecay': tf.keras.optimizers.schedules.PolynomialDecay,
    'InverseTimeDecay': tf.keras.optimizers.schedules.InverseTimeDecay,
    'CosineDecay': tf.keras.optimizers.schedules.CosineDecay,
    'CosineDecayRestarts': tf.keras.optimizers.schedules.CosineDecayRestarts,
}

def get_lr_scheduler(**kwargs):
    '''
    Get learning rate scheduler from `SCHEDULER_DICT`
    '''
    assert kwargs['name'] in SCHEDULER_DICT, f"Scheduler {kwargs['name']} not supported"
    return SCHEDULER_DICT[kwargs['name']](**kwargs)
