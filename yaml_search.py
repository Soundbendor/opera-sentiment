import yaml
import os
from ENV import Data_PATH
from yamlhelp import safe_read_yaml

def search_song_info(search_for, value, counter_select=False, looking_for_empty = False): # -> list of song_id
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
                            print("the file which is missing the key is: " + yaml_path)
                            raise KeyError("key error, please check the yaml template to make sure the key is correct")
                    
                    if looking_for_empty:
                        if str(meta) == "":
                            res.append(meta_og["song_id"])
                    else:
                        if not counter_select:
                            if str(value) in str(meta):
                                res.append(meta_og["song_id"])
                        else:
                            if str(value) not in str(meta):
                                res.append(meta_og["song_id"])
    else:
        raise ValueError("this function is only for searching for songs, use search_recording_info and \"files\" as the first search key for searching for files")
    # sort the list in incresing order
    res.sort()
    return res

def check_Fan3Chuan4(): # -> list of song_id: wav name
    # search for e.g. singer info or if_a_cappella
    res = []
    for root, dirs, files in os.walk(Data_PATH):
        for file in files:
            if file.endswith(".yaml"):
                yaml_path = os.path.join(root, file)
                meta = safe_read_yaml(yaml_path)
                if meta['singing_type']['role'] == 'sheng':
                    gender_flag = 'mal'
                elif meta['singing_type']['role'] == 'dan':
                    gender_flag = 'fem'
                else:
                    print("this file is missing role type: " + yaml_path)
                    continue
                if "files" in meta:
                    for wav in meta["files"]:
                        if meta["files"][wav]["singer"]["bio_gender"] != gender_flag:
                            res.append((meta["song_id"], wav))
    return res

if __name__ == "__main__":
    search_for = ["singing_type", "role"]
    value = "TBD"

    # res = search_song_info(search_for, value, counter_select=False)
    # if len(res) == 0:
    #     print("no result found")
    
    res = check_Fan3Chuan4()
    if len(res) == 0:
        print("no result found")
    
    for id in res:
        file_path = os.path.join(Data_PATH, "ch", str(id[0]), "metadata.yaml")
        if os.path.exists(file_path):
            yaml_path = file_path
        else:
            yaml_path = os.path.join(Data_PATH, "we", str(id[0]), "metadata.yaml")
        meta = safe_read_yaml(yaml_path)
        print("--id: " + str(id) + "--")
        print(meta["title"]["original"])
        print(meta["title"]["phonetic"])
        print("\n")