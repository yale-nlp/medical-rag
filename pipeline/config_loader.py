import logging
import json
from pathlib import Path

def setup_logging(debug_mode: bool):
    """
    Configures the root logger for the application.
    - INFO level for normal operation (shows progress).
    - DEBUG level for verbose output (--debug flag).
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress noisy logs from underlying libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

def load_config(config_path: str = "configs.json") -> dict:
    """Loads the main JSON configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Main config file not found at {config_file}. Please ensure it exists.")
    try:
        with open(config_file, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding main config file {config_file}. Please ensure it's valid JSON.")