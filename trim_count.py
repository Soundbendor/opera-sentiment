# get basic info for this dataset:

import os
from ENV import target_second, Unified_PATH, Trimmed_PATH

# get how many pieces in each song
song_id_to_trimed_count = {
    "ch": {},
    "we": {}
}

def get_trim_count():
    for lan in ["ch", "we"]:
        for song_id in os.listdir(os.path.join(Trimmed_PATH, lan)):
            if song_id.isdigit():
                song_id = int(song_id)
                song_id_to_trimed_count[lan][song_id] = 0
                song_id_path = os.path.join(Trimmed_PATH, lan, str(song_id))
                for path, dirs, files in os.walk(song_id_path):
                    for file in files:
                        if file.endswith(".wav"):
                            song_id_to_trimed_count[lan][song_id] += 1

get_trim_count()