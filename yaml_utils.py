
import yaml
import numpy as np
from skopt.space import Integer, Real

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(i) for i in obj]
    return obj

def space_dim_to_dict(dim):
    if isinstance(dim, Integer):
        return {'type': 'int', 'low': dim.low, 'high': dim.high, 'name': dim.name}
    elif isinstance(dim, Real):
        return {'type': 'float', 'low': dim.low, 'high': dim.high, 'name': dim.name}
    else:
        raise ValueError(f"Unsupported dimension type: {type(dim)}")

def dict_to_space_dim(dim_dict):
    if dim_dict['type'] == 'int':
        return Integer(dim_dict['low'], dim_dict['high'], name=dim_dict['name'])
    elif dim_dict['type'] == 'float':
        return Real(dim_dict['low'], dim_dict['high'], name=dim_dict['name'])
    else:
        raise ValueError(f"Unsupported dimension type: {dim_dict['type']}")

def save_results(filename, data):
    serializable_data = {
            k: ndarray_to_list(v) for k, v in data.items()
    }
    if 'space' in serializable_data:
        serializable_data['space'] = [space_dim_to_dict(dim) for dim in serializable_data['space']]
    with open(filename, 'w') as f:
        yaml.dump(serializable_data, f)

def load_results(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    if 'space' in data:
        data['space'] = [dict_to_space_dim(dim_dict) for dim_dict in data['space']]
    return data
