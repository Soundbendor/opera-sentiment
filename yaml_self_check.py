# run this scrip to self check if the yaml metadata files have any bugs in it

import os
import yaml
from PATH import Data_PATH

yaml_answer_path = "metadata_answer.yaml"

with open(yaml_answer_path,"r") as f:
    yaml_answer = yaml.safe_load(f)

for root, dirs, files in os.walk(Data_PATH):
    for file in files:
        if file.endswith(".yaml"):
            yaml_path = os.path.join(root, file)
            with open(yaml_path,"r") as f:
                meta = yaml.safe_load(f)
            flag = True
            if yaml_answer.keys() != meta.keys():
                flag = False
                print("ERROR: {} has different keys".format(yaml_path))
            else:
                # specific check if if_a_cappella is bool
                for wav in meta["files"]:
                    if type(meta["files"][wav]["info"]["if_a_cappella"]) != bool:
                        flag = False
                        print("ERROR: {} has non-bool if_a_cappella value(s)".format(yaml_path))
                # check if the song_size is correct
                if meta["song_size"] != len(meta["files"]):
                    flag = False
                    print("ERROR: {} has wrong song_size".format(yaml_path))

            if flag:
                print("{} is good".format(yaml_path))