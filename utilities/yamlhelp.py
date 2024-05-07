import os
import yaml

def safe_read_yaml(yaml_file):
    if not os.path.exists(yaml_file):
        raise FileNotFoundError("File not found: {}".format(yaml_file))
    if not yaml_file.endswith('.yaml'):
        raise ValueError("File is not a yaml file: {}".format(yaml_file))
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

def safe_update_yaml(yaml_file, data):
    if not os.path.exists(yaml_file):
        raise FileNotFoundError("File not found: {}, this function is only for update, not create".format(yaml_file))
    if not yaml_file.endswith('.yaml'):
        raise ValueError("File is not a yaml file: {}".format(yaml_file))
    with open(yaml_file, 'w') as stream:
        try:
            yaml.dump(data, stream)
        except yaml.YAMLError as exc:
            raise exc