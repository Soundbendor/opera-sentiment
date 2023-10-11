import os
import yaml

path = "operadataset2023"

existing_title = []
for root, dirs, files in os.walk(path):
    if not dirs:
        meta_file = os.path.join(root, "metadata.yaml")
        # if meta_file exist:
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = yaml.safe_load(f)
                if meta["singing_type"]["singing"] != "jingju":
                    print("{} is ".format(root)+meta["singing_type"]["singing"])
                # if "TBD" in meta["singing_type"]["singing"]:
                #     print("{} is ".format(root), meta["singing_type"]["singing"])
        # else:
        #     print("No metadata.yaml in {}".format(root))

