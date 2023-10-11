import yaml
import sys

yaml_name = "metadata_template.yaml"
# take an argument for how many files there are， otherwise 1

try: how_many_files = int(sys.argv[1])
except: how_many_files = 1

meta = {
    "emotion":[],
    "emotion_binary":-1,
    "files":{},
    "language":"",
    "lyric":{
        "english": "",
        "original": "",
        "phonetic": "",
    },
    "scene":{
        "english": "",
        "original": "",
        "phonetic": "",
    },
    "singing_type":{
        "role": "",
        "singing": "",
    },
    "song_dir": "",
    "song_id": -1,
    "song_size": how_many_files,
    "title":{
        "english": "",
        "original": "",
        "phonetic": "",
    },
    "wiki":""
}

for i in range(how_many_files):
    if i < 10:
        file_name = "wav0"+str(i)
    else:
        file_name = "wav"+str(i)
    meta["files"][file_name] = {
        "file_dir": "",
        "info":{
            "bit_rate": -1,
            "channels": -1,
            "duration": -1,
            "if_a_cappella": "",
            "sample_rate": -1,
        },
        "singer": {
            "bio_gender": "",
            "id": "",
            "level": "",
            "name": "",
        }
    }

with open(yaml_name, "w") as f:
    yaml.safe_dump(meta, f, allow_unicode=False)
