import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import tensorflow as tf

from archs import get_model
from utils import ConfigParser, logging, TfdataPipeline, get_metric_fn

def get_args():
    parser = argparse.ArgumentParser(description="Script to train image restoration models.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    return args

def test_model(save_results=False):
    args = get_args()
    logging.info(f"Config file loaded from {args.config}")
    config_parser_obj = ConfigParser(args.config)
    logging.info(f"Config file parsed successfully..")
    config_raw = config_parser_obj.get_config()
    try:
        experiment_name              = config_raw['name']
        seed                         = config_raw['manual_seed']
        dataset_config               = config_parser_obj.get_dataset_config()
        dataset_config['batch_size'] = 1
    except KeyError as e:
        logging.error(f"KeyError: {e} not found in config file.")
    except Exception as e:
        logging.error(f"Exception: {e}.")

    tf.random.set_seed(seed)

    # Instantiate tf.data pipeline
    logging.info("Initializing test tf.data pipeline..")
    tfdataset           = TfdataPipeline(**dataset_config)
    test_dataset        = tfdataset.data_loader(dataset_type='test', do_augment=False)
    
    logging.info("Creating model..")
    model_name          = config_parser_obj.get_model_name().lower()
    model_config        = config_parser_obj.get_model_config()
    model               = get_model(model_name, **model_config)

    model_path = config_parser_obj.get_model_save_config()['checkpoint_dir']
    model_path = os.path.join(model_path, experiment_name)
    
    logging.info(f"Restoring model from {model_path}..")
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    logging.info(f"Model restored from {model_path}")
    
    metrics_fn_dict        = get_metric_fn(config_parser_obj.get_metric_functions())
    logging.info(f"Selected metrics: {metrics_fn_dict.keys()}")

    test_results = {}

    for i, (x, y) in enumerate(test_dataset.take(5)):
        y_pred = model(x)
        print(f'test {i+1}:')
        for metric_name, metric_fn in metrics_fn_dict.items():
            metric_value = metric_fn(y, y_pred)
            if metric_name not in test_results.keys():
                test_results[metric_name] = 0
            test_results[metric_name] += metric_value
            print(f"{metric_name}: {metric_value}")
        if save_results:
            x, y, y_pred = tf.squeeze(x), tf.squeeze(y), tf.squeeze(y_pred)
            combined_image = tf.concat([x, y, y_pred], axis=1)
            tf.keras.preprocessing.image.save_img(f"outputs/test_{i+1}.png", combined_image)
    test_results["count"] = i+1

    # Average the metrics
    for metric_name in test_results.keys():
        test_results[metric_name] /= test_results["count"]
    test_results.pop("count")
    
    logging.info(f"Average test results:")
    for metric_name, metric_value in test_results.items():
        logging.info(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    test_model(save_results=True)