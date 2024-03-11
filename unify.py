import os
import shutil
import subprocess
from ENV import Data_PATH, Unified_PATH
from Profiler import Profiler
import soxtool as st
import yaml

def create_unified_directory(src, dest, exclude_extensions=['.wav']):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest):
        os.makedirs(dest)

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dest, item)

            # Check if the item is a directory
            if os.path.isdir(s):
                create_unified_directory(s, d, exclude_extensions)
            else:
                # Check if the file extension is in the exclusion list
                _, extension = os.path.splitext(item)
                if extension.lower() not in (exclude_extensions or []):
                    shutil.copy2(s, d)
    else:
        print('Destination directory already exists. Double-check what you really want to do.')

def unify(source_wav): # unify the data
    destination_wav = source_wav.replace(Data_PATH, Unified_PATH)
    CMD_convert = ['sox', source_wav, '-r', '16000', '-c', '1', '-b', '16', destination_wav]
    update_yaml(destination_wav)
    subprocess.run(CMD_convert)
    if st.get_SR(destination_wav) != 16000:
        print(destination_wav)
    if st.get_bit(destination_wav) != 16:
        print(destination_wav)
    if st.get_channel(destination_wav) != 1:
        print(destination_wav)

def update_yaml(destination_wav):
    wav_id = destination_wav.split("/")[3].replace(".wav", "")

    yaml_path = ""
    for yaml_path_parts in destination_wav.split("/")[0:-1]:
        yaml_path+=yaml_path_parts+"/"
    yaml_path+="metadata.yaml"

    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    meta["files"][wav_id]["info"]["sample_rate"] = 16000
    meta["files"][wav_id]["info"]["channel_number"] = 1
    meta["files"][wav_id]["info"]["bit_rate"] = 16
    # write back to yaml
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, allow_unicode=False)

if __name__ == '__main__':
    src_directory = Data_PATH
    dest_directory = Unified_PATH

    create_unified_directory(src_directory, dest_directory)

    # get profile in order to get wave info
    source_profile = Profiler(src_directory)
    
    # unify chinese opera
    for sub_wav in source_profile.wav_info['ch']:
        wav = os.path.join(src_directory, sub_wav)
        unify(wav)
    # unify western opera
    for sub_wav in source_profile.wav_info['we']:
        wav = os.path.join(src_directory, sub_wav)
        unify(wav)
    
    # check the new profile
    unified_profile = Profiler(dest_directory)
    unified_profile.full_profile(if_print_profile=True)
