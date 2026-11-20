import yaml
from pathlib import Path


def load_config(path):
    with open(path,encoding='utf8') as handle:
        return yaml.safe_load(handle)


def config_hash(cfg):
    import json,hashlib
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
