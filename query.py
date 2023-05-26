import os
import yaml

path = "operadataset2023"

existing_title = []
for root, dirs, files in os.walk(path):
    if not dirs:
        meta_file = os.path.join(root, "metadata.yaml")
        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)
            # TODO