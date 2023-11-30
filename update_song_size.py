# add the ymal entry "song_size": represent how many audio in this one single song,
# but before that, need to change the new song names first

import os
import yaml
import sys

Data_PATH = "operadataset2023"

def update_single(song_path):
    if len(song_path.split("/")) != 3:
        PATH = os.path.join(Data_PATH, song_path)
    else: 
        PATH = song_path
    meta_file = os.path.join(PATH, "metadata.yaml")

    # check if meta file exists
    if not os.path.exists(meta_file):
        print("meta file " + meta_file + " doesn't exist, check your input")
        exit()

    with open(meta_file,"r") as f:
        meta = yaml.safe_load(f)
        files = meta["files"]
        song_size = len(files)
        old_value = meta["song_size"]
        if old_value == song_size:
            print("song size for " + song_path + " is already correct")
        else: 
            print("song size update from " + str(old_value) + " to " + str(song_size) + " for " + song_path)
            meta["song_size"] = song_size

        # write back to yaml
        with open(meta_file, "w") as f:
            yaml.safe_dump(meta, f, allow_unicode=False)

def update_all():
    for root, dirs, files in os.walk(Data_PATH):
        for file in files:
            if file.endswith(".yaml"):
                # print(root)
                update_single(root)

if __name__ == "__main__":
    input_path = sys.argv[1] # eg: ch/9
    if input_path == "all":
        update_all()
    else:
        update_single(input_path)



