# add the ymal entry "song_size": represent how many audio in this one single song,
# but before that, need to change the new song names first

import os
import yaml
import sys

song_id = sys.argv[1]

song_id = song_id
song_PATH = "ch/{}".format(song_id)
Data_PATH = "operadataset2023"
PATH = os.path.join(Data_PATH, song_PATH)
meta_file = os.path.join(PATH, "metadata.yaml")

with open(meta_file,"r") as f:
    meta = yaml.safe_load(f)
    files = meta["files"]
    song_size = len(files)
    meta["song_size"] = song_size

# write back to yaml
with open(meta_file, "w") as f:
    yaml.safe_dump(meta, f, allow_unicode=False)