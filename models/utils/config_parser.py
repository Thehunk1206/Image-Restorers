import os
import sys
import yaml

class ConfigParser:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None

        if not os.path.exists(self.config_file):
            print("Config file does not exist")
            sys.exit(1)

        with open(self.config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    def get_config(self):
        return self.config
    
    def get_model_name(self):
        try:
            return self.config['model']['model_name']
        except KeyError:
            print("Model name not found")
    
    def get_model_config(self):
        try:
            return self.config['model']['model_params']
        except KeyError:
            print("Model config not found")
    
    def get_dataset_config(self):
        try:
            return self.config['datasets']
        except KeyError:
            print("Dataset config not found")
    
    def get_train_config(self):
        try:
            return self.config['train']
        except KeyError:
            print("Train config not found")
    
    def get_test_config(self):
        try:
            return self.config['test']
        except KeyError:
            print("Test config not found")

    def get_keys(self):
        return self.config.keys()

if __name__ == "__main__":
    config_parser = ConfigParser("/home/developer/workspace/configs/dummy_config.yml")
    # print(config_parser.get_config())
    print(config_parser.get_model_config())
    print(config_parser.get_dataset_config())
    print(config_parser.get_train_config())
    print(config_parser.get_test_config())