import os
import yaml

def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'config',
        'global.yaml'
    )
    # Open with UTF-8 encoding
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)