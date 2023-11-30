import yaml
import os
from ENV import Data_PATH
from yamlhelp import safe_read_yaml

search_for = ["song_size"]
value = 2

for root, dirs, files in os.walk(Data_PATH):
    for file in files:
        if file.endswith(".yaml"):
            yaml_path = os.path.join(root, file)
            meta = safe_read_yaml(yaml_path)
            if search_for[0] != "files": # it will be different for looking for files because there will be multiple wav under files
                for search in search_for:
                    if search in meta:
                        meta = meta[search]
                    else:
                        raise KeyError("key error, please check the yaml template to make sure the key is correct")
            else:
                # TODO
                raise NotImplementedError("search for \"files\" is not implemented yet")
            if str(value) in str(meta):
                print(yaml_path)