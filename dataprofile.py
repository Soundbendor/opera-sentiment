import os
import yaml
from PATH import Data_PATH

wav_info = {
    "ch": {},
    "we": {}
}
song_size = {
    "ch": 0,
    "we": 0
}

singing_types = {
    "ch": {},
    "we": {}
}

jj_only_roles = {}

# Song Level Traverse
for root, dirs, files in os.walk(Data_PATH):
    for file in files:
        # we only need to get info from yaml files
        if file.endswith(".yaml"):
            yaml_path = os.path.join(root, file)
            if "ch" in yaml_path:
                lan = "ch"
                song_size["ch"]+=1
            else:
                lan = "we"
                song_size["we"]+=1
            with open(yaml_path,"r") as f:
                meta = yaml.safe_load(f)
            
            singing_type = meta["singing_type"]["singing"]
            singing_types[lan][singing_type] = singing_types[lan].get(singing_type, 0) + 1

            if singing_type == "jingju":
                role = meta["singing_type"]["role"]
                jj_only_roles[role] = jj_only_roles.get(role, 0) + 1

            # get an info dict for each wav file
            for wav in meta["files"]:
                file_name = meta["files"][wav]["file_dir"]
                if file_name not in wav_info[lan]:
                    wav_info[lan][file_name] = {}
                wav_info[lan][file_name]["info"] = meta["files"][wav]["info"]
                wav_info[lan][file_name]["singer"] = meta["files"][wav]["singer"]

recording_size = {
    "ch": 0,
    "we": 0
}

a_cappellas = {
    "ch": {
        "true": 0,
        "false": 0
    },
    "we": {
        "true": 0,
        "false": 0
    }
}

bit_rates = {
    "ch": {},
    "we": {}
}

channel_number_s = {
    "ch": {},
    "we": {}
}

sample_rates = {
    "ch": {},
    "we": {}
}

bio_genders = {
    "ch": {},
    "we": {}
}

singer_ids = {
    "ch": {},
    "we": {}
}

singer_levels = {
    "ch": {},
    "we": {}
}

singer_names = {
    "ch": {},
    "we": {}
}

len_sum = {
    "ch": 0,
    "we": 0
}

len_max = {
    "ch": -1,
    "we": -1
}

len_min = {
    "ch": -1,
    "we": -1
}

# Recording Level Traverse
for lan in wav_info:
    for wav in wav_info[lan]:
        recording_size[lan] += 1
        if wav_info[lan][wav]["info"]["if_a_cappella"] == True:
            a_cappellas[lan]["true"] += 1
        elif wav_info[lan][wav]["info"]["if_a_cappella"] == False:
            a_cappellas[lan]["false"] += 1
        else:
            print("ERROR: {} has non-bool if_a_cappella value(s)".format(wav))
        
        bit_rate = wav_info[lan][wav]["info"]["bit_rate"]
        bit_rates[lan][bit_rate] = bit_rates[lan].get(bit_rate, 0) + 1
        
        channel_number = wav_info[lan][wav]["info"]["channel_number"]
        channel_number_s[lan][channel_number] = channel_number_s[lan].get(channel_number, 0) + 1

        sample_rate = wav_info[lan][wav]["info"]["sample_rate"]
        sample_rates[lan][sample_rate] = sample_rates[lan].get(sample_rate, 0) + 1

        bio_gender = wav_info[lan][wav]["singer"]["bio_gender"]
        bio_genders[lan][bio_gender] = bio_genders[lan].get(bio_gender, 0) + 1

        singer_id = wav_info[lan][wav]["singer"]["id"]
        singer_ids[lan][singer_id] = singer_ids[lan].get(singer_id, 0) + 1

        singer_level = wav_info[lan][wav]["singer"]["level"]
        singer_levels[lan][singer_level] = singer_levels[lan].get(singer_level, 0) + 1

        singer_name = wav_info[lan][wav]["singer"]["name"]
        singer_names[lan][singer_name] = singer_names[lan].get(singer_name, 0) + 1      

        len = wav_info[lan][wav]["info"]["duration"]
        if len_max[lan] == -1 or len > len_max[lan]:
            len_max[lan] = len
        if len_min[lan] == -1 or len < len_min[lan]:
            len_min[lan] = len
        len_sum[lan]+=len

len_mean = {
    "ch": len_sum["ch"]/recording_size["ch"],
    "we": len_sum["we"]/recording_size["we"]
}

print("singing_types: {}".format(singing_types))
print("roles in jingju: {}".format(jj_only_roles))
print("if_a_cappellas: {}".format(a_cappellas))
print("song_size: {}".format(song_size))
print("number of recording: {}".format(recording_size))
print("max_len: {}".format(len_max))
print("min_len: {}".format(len_min))
print("mean_len: {}".format(len_mean))
print("bit_rates: {}".format(bit_rates))
print("channel_number_s: {}".format(channel_number_s))
print("sample_rates: {}".format(sample_rates))
print("bio_gender: {}".format(bio_genders))
print("singer_id: {}".format(singer_ids))
print("singer_level: {}".format(singer_levels))
print("singer_name: {}".format(singer_names))

