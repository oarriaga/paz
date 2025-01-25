import os
import json
from keras.utils import get_file


def get_data_json(version):
    if version in ['latest', 'arcagi', 'aa922be']:
        return json.load(open(f"{os.path.dirname(__file__)}/data/arcagi_aa922be.json"))
    
    elif version in ['kaggle', 'kaggle2024']:
        return json.load(open(f"{os.path.dirname(__file__)}/data/kaggle2024.json"))
    
    elif version in ['arc', 'kaggle2019']:
        return json.load(open(f"{os.path.dirname(__file__)}/data/arc1.json"))
    
    else:
        raise ValueError(f"Unknown ARC dataset version: {version}")


def load(path, split)
