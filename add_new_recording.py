import os
import yaml
import soxtool as st
import sys

# When you want to add a new recording to a song:
# Add the path to yaml manually
# run this script
# fix human-needed part: if_a_cappella, bio_gender, id, ... ...
# if they are all the same and predicatble, then change these variables:
singer_level = "mixed"
singer_name = "mixed level group"
if_a_cappella = True

# Finally run update_song_size.py to update the song size

song_id = sys.argv[1]

song_PATH = "ch/{}".format(song_id)

Data_PATH = "operadataset2023"
PATH = os.path.join(Data_PATH, song_PATH)
meta_file = os.path.join(PATH, "metadata.yaml")
with open(meta_file,"r") as f:
    meta = yaml.safe_load(f)
    files = meta["files"]
    for wav in files:
        wav_detail = files[wav]
        if "info" not in wav_detail.keys():
            wav_path = os.path.join(Data_PATH, wav_detail["file_dir"])
            # get infos:
            # get duration
            duration = st.get_duration(wav_path)
            # get channel
            channel = st.get_channel(wav_path)
            # get sample rate
            SR = st.get_SR(wav_path)
            # get bit rate
            bit = st.get_bit(wav_path)
            # if_a_capella need to determine by human
            # write those in:
            wav_detail["info"] = {"duration": duration, 
                                  "channel_number": channel, 
                                  "sample_rate": SR, 
                                  "bit_rate": bit, 
                                  "if_a_cappella": if_a_cappella}
        if "singer" not in wav_detail.keys():
            wav_detail["singer"] = {"bio_gender": "unknown",
                                    "id": "unknown",
                                    "level": singer_level,
                                    "name": singer_name}

# write back to yaml
with open(meta_file, "w") as f:
    yaml.safe_dump(meta, f, allow_unicode=False)