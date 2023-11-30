import yaml
import os
from ENV import Data_PATH
from yamlhelp import safe_read_yaml

def search_song_info(search_for, value): # -> list of song_id
    res = []
    # it will be different for looking for files because there will be multiple wav under files
    if search_for[0] != "files":
        for root, dirs, files in os.walk(Data_PATH):
            for file in files:
                if file.endswith(".yaml"):
                    yaml_path = os.path.join(root, file)
                    meta_og = safe_read_yaml(yaml_path)
                    meta = safe_read_yaml(yaml_path)
                    
                    for search in search_for:
                        if search in meta:
                            meta = meta[search]
                        else:
                            raise KeyError("key error, please check the yaml template to make sure the key is correct")
                    if str(value) in str(meta):
                        res.append(meta_og["song_id"])
    else:
        raise ValueError("this function is only for searching for songs, use search_recording_info and \"files\" as the first search key for searching for files")
    # sort the list in incresing order
    res.sort()
    return res

def search_recording_info(): # -> list of song_id: wav name
    return
    # TODO: search for e.g. singer info or if_a_cappella

if __name__ == "__main__":
    search_for = ["emotion_binary"]
    value = -1
    res = search_song_info(search_for, value)
    for id in res:
        yaml_path = os.path.join(Data_PATH, "ch", str(id), "metadata.yaml")
        meta = safe_read_yaml(yaml_path)
        print("--id: " + str(id) + "--")
        print(meta["title"]["original"])
        print(meta["scene"]["original"])
        print("\n")