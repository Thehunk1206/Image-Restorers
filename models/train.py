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
import argparse
import time
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf

from archs import get_model
from utils import (
    ConfigParser,
    TfdataPipeline,
    get_loss_fn,
    get_metric_fn,
    get_lr_scheduler,
    logging
)

from image_restoration_model import ImageRestorationModel

def get_args():
    parser = argparse.ArgumentParser(description="Script to train image restoration models.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    return args

def train():
    args = get_args()
    logging.info(f"Config file loaded from {args.config}")
    config_parser_obj = ConfigParser(args.config)
    config_raw = config_parser_obj.get_config()
    logging.info(f"Config file parsed successfully..")
    
    # get some config params
    try:
        experiment_name     = config_raw['name']
        model_type          = config_raw['model_type']
        seed                = config_raw['manual_seed']
        model_save_config   = config_parser_obj.get_model_save_config()
        train_config        = config_parser_obj.get_train_config()
        dataset_config      = config_parser_obj.get_dataset_config()
    except KeyError as e:
        logging.error(f"KeyError: {e} not found in config file.")
    except Exception as e:
        logging.error(f"Exception: {e}.")
    
    tf.random.set_seed(seed)

    # Initialize tf summary writer
    logging.info(f"Initializing tf summary writer..")
    tb_config           = config_parser_obj.get_tb_logger_config()
    logs_dir            = f"{tb_config['log_dir']}/{experiment_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train_writer        = tf.summary.create_file_writer(logs_dir + '/train')
    val_writer          = tf.summary.create_file_writer(logs_dir + '/val')

    # Instantiate tf.data pipeline
    logging.info("Initializing tf.data pipeline..")
    tfdataset           = TfdataPipeline(**dataset_config)
    train_data          = tfdataset.data_loader(dataset_type='train', do_augment=True)
    val_data            = tfdataset.data_loader(dataset_type='valid', do_augment=False)

    # Instantiate optimizer and scheduler
    logging.info("Initializing optimizer and scheduler..")
    optimizer_config    = config_parser_obj.get_optimizer_config()
    scheduler_config    = config_parser_obj.get_scheduler_config()

    lr_scheduler_fn     = get_lr_scheduler(**scheduler_config)
    optimizer           = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_fn, **optimizer_config)

    # Instantiate model
    logging.info("Initializing model..")
    model_name          = config_parser_obj.get_model_name()
    model_config        = config_parser_obj.get_model_config()
    model               = get_model(model_name, **model_config)
    model               = ImageRestorationModel(restore_model=model)
    logging.info(f"{model.name} model created successfully..")
    logging.info(f"Model summary:")
    model.summary()

    # Creat loss functions dict
    loss_fn_dict        = {}
    loss_fn             = config_parser_obj.get_loss_fucntions()
    for loss_name, loss_config in loss_fn.items():
        loss = get_loss_fn(loss_name)
        loss_fn_dict[loss_name] = loss(**loss_config)
    logging.info(f"Selected losses: {loss_fn_dict.keys()}")

    # Create metrics dict
    metrics_fn_dict        = get_metric_fn(config_parser_obj.get_metric_functions())
    logging.info(f"Selected metrics: {metrics_fn_dict.keys()}")

    # Compile model
    logging.info(f"Compiling model..")
    model.compile(
        optimizer   = optimizer,
        loss        = loss_fn_dict,
        metrics_fn  = metrics_fn_dict
    )

    # Train model
    logging.info(f"\nTraining with following configurations: ")
    for key, value in config_raw.items():
        logging.info(f"{key}: {value}")

if __name__ == '__main__':
    train()