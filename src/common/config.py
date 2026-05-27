import yaml
import os

def load_config() -> dict:
    """加载配置文件"""
    # Get the directory of the current script (src/common)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the config file
    config_path = os.path.join(script_dir, '..', '..', 'config', 'config.yaml')
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()