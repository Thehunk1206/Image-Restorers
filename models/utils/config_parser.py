import os
import sys
import yaml
try:
    from logger import logging
except:
    from utils.logger import logging

class ConfigParser:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None

        if not os.path.exists(self.config_file):
            logging.error(f"{__name__} - Config file {self.config_file} does not exist")
            sys.exit(1)

        with open(self.config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.error(f"{__name__} - Error in config file-{exc}")
                sys.exit(1)

    def get_config(self):
        return self.config
    
    def get_dataset_config(self):
        try:
            return self.config['datasets']
        except KeyError:
            logging.error(f"{__name__} - Dataset config not found")

    def get_model_name(self):
        try:
            return self.config['model']['name']
        except KeyError:
            logging.error(f"{__name__} - Model name not found")
    
    def get_model_config(self):
        try:
            return self.config['model']['model_params']
        except KeyError:
            logging.error(f"{__name__} - Model config not found")
    
    # train config
    def get_train_config(self):
        try:
            return self.config['train']
        except KeyError:
            logging.error(f"{__name__} - Train config not found")
    
    def get_optimizer_config(self):
        try:
            return self.get_train_config()['optimizer']  
        except KeyError:
            logging.error(f"{__name__} - Optimizer config not found")
    
    def get_scheduler_config(self):
        try:
            return self.get_train_config()['scheduler']
        except KeyError:
            logging.error(f"{__name__} - Scheduler config not found")
    
    def get_loss_fucntions(self):
        try:
            return self.get_train_config()['losses']
        except KeyError:
            logging.error(f"{__name__} - Loss function not found")
    
    def get_metric_functions(self):
        try:
            return self.get_train_config()['metrics']
        except KeyError:
            logging.error(f"{__name__} - Metric functions not found")
    
    def get_test_config(self):
        try:
            return self.config['test']
        except KeyError:
            logging.error(f"{__name__} - Test config not found")
    
    def get_model_save_config(self):
        try:
            return self.config['model_save']
        except KeyError:
            logging.error(f"{__name__} - Model save config not found")
    
    def get_tb_logger_config(self):
        try:
            return self.config['tb_logger']
        except KeyError:
            logging.error(f"{__name__} - Tensorboard logger config not found")

    def get_keys(self):
        return self.config.keys()

if __name__ == "__main__":
    config_parser = ConfigParser("configs/nafnet-width16-config.yaml")
    # print(config_parser.get_train_config())
    logging.info(config_parser.get_optimizer_config())
    logging.info(config_parser.get_scheduler_config())
    logging.info(config_parser.get_loss_fucntions())
    logging.info(config_parser.get_metric_functions())