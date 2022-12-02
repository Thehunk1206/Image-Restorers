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

# TODO: Add support for training pretrained models
# TODO: Add support for multiple GPUs
# TODO: Add support for mixed precision training

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
        dataset_config      = config_parser_obj.get_dataset_config()
        train_config        = config_parser_obj.get_train_config()
        model_save_config   = config_parser_obj.get_model_save_config()
        tf_logger_config    = config_parser_obj.get_tb_logger_config()
    except KeyError as e:
        logging.error(f"KeyError: {e} not found in config file.")
    except Exception as e:
        logging.error(f"Exception: {e}.")
    
    tf.random.set_seed(seed)

    # Initialize tf summary writer
    logging.info(f"Initializing tf summary writer..")
    if not os.path.exists(tf_logger_config['log_dir']):
        os.makedirs(tf_logger_config['log_dir'])
    logs_dir            = f"{tf_logger_config['log_dir']}/{experiment_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train_writer        = tf.summary.create_file_writer(logs_dir + '/train')
    val_writer          = tf.summary.create_file_writer(logs_dir + '/val')

    # Instantiate tf.data pipeline
    logging.info("Initializing tf.data pipeline..")
    tfdataset           = TfdataPipeline(**dataset_config)
    train_data          = tfdataset.data_loader(dataset_type='train', do_augment=True)
    val_data            = tfdataset.data_loader(dataset_type='valid', do_augment=True)

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
    logging.info(f"Compiling model..\n")
    model.compile(
        optimizer   = optimizer,
        loss        = loss_fn_dict,
        metrics_fn  = metrics_fn_dict
    )

    # Train model
    logging.info(f"Training with following configurations: ")
    for key, value in config_raw.items():
        if isinstance(value, dict):
            for k, v in value.items():
                logging.info(f"{k}: {v}")
        else:
            logging.info(f"{key}: {value}")
    
    logging.info(f"Training started..")

    for epoch in range(1,train_config['epoch']+1):
        start_time = time.time()

        for (train_input_image, train_target_image) in tqdm(train_data, unit='steps', desc=f"Epoch {epoch} - Training", colour='red'):
            train_step_results = model.train_step(train_input_image, train_target_image)
            
            with train_writer.as_default():
                tf.summary.scalar(name='lr', data=optimizer.learning_rate, step=optimizer.iterations)
        
        for (val_input_image, val_target_image) in tqdm(val_data, unit='steps', desc=f"Epoch {epoch} - Validation", colour='green'):
            val_step_results = model.test_step(val_input_image, val_target_image)
        
        eta        = round(((time.time() - start_time)/60.0) * (train_config['epoch'] - epoch), 2)
        epoch_time = round((time.time() - start_time)/60.0, 2)
        logging.info(f"Epoch {epoch} completed in {epoch_time} mins. ETA: {eta} mins.\n")
        logging.info(f"Train results: {train_step_results}\n")
        logging.info(f"Validation results: {val_step_results}\n")

        # write to tensorboard 
        logging.info(f"Writing to tensorboard..\n")
        with train_writer.as_default():
            for name, data in train_step_results.items():
                tf.summary.scalar(name, data, step=epoch)

        with val_writer.as_default():
            for name, data in val_step_results.items():
                tf.summary.scalar(name, data, step=epoch)
        
        if tf_logger_config["log_image"]: # log image
            pred = model.restore_model(val_input_image)
            with val_writer.as_default():
                tf.summary.image("Input", val_input_image, step=epoch, max_outputs=3, description='Input image')
                tf.summary.image("Target", val_target_image, step=epoch, max_outputs=3, description='Target image')
                tf.summary.image("Predicted", pred, step=epoch, max_outputs=3, description='Predicted image')

        # Save model
        if epoch % model_save_config['frequency']:
            logging.info(f"Saving model..")
            if not os.path.exists(model_save_config['checkpoint_dir']):
                os.makedirs(model_save_config['checkpoint_dir'])
            
            model_path = f'{model_save_config["checkpoint_dir"]}/{experiment_name}_{epoch}'
            model.save(
                model_path,
                save_format= model_save_config['save_format'],
                save_only_weights= model_save_config['save_only_weights']
            )
            logging.info(f"Model saved at {model_path}")

if __name__ == '__main__':
    train()