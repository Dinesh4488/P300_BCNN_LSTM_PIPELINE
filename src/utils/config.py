import yaml

def load_config(config_path):
    """
    Loads a YAML configuration file.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
