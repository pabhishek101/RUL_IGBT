import yaml

def load_config(config_file="code\config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

