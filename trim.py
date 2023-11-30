import os
from ENV import Unified_PATH, target_second, Trimmed_PATH

import shutil
import subprocess
from dataprofile import Profiler
import sox

# create a new directory for the trimmed data
def create_trimmed_directory(src, dest):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest):
        os.makedirs(dest)

        # Iterate over files and subdirectories in the source directory
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dest, item)

            if os.path.isdir(s):
                # Recursively copy subdirectories
                create_trimmed_directory(s, d)
            else:
                # Check if the file has a .wav extension
                if s.endswith('.wav'):
                    # Create a folder with the file name (excluding extension)
                    folder_name = os.path.splitext(item)[0]
                    folder_path = os.path.join(dest, folder_name)

                    # Create the folder if it doesn't exist
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                else:
                    # Copy other files directly
                    shutil.copy2(s, d)
    else:
        print('Destination directory already exists. Double-check what you really want to do.')

def trim_batch(src_directory, dest_directory, lan):
    size = target_second
    source_profiler = Profiler(src_directory)
    def trim_lan(lan):
        for sub_wav in source_profiler.wav_info[lan]:
            src_wav = os.path.join(src_directory, sub_wav)
            det_wav_temp = os.path.join(os.path.join(dest_directory, sub_wav).split(".")[0], sub_wav.split("/")[-1])
            # add an underscore to the end of the file name
            det_wav = det_wav_temp.split(".")[0] + "_.wav"
            trim(src_wav, det_wav, size)

    if lan == "ch" or lan == "all":
        trim_lan("ch")
    if lan == "we" or lan == "all":
        trim_lan("we")

def trim(src, dest, size):
    CMD_split = ['sox', src, dest, 'trim', '0', str(size), ':', 'newfile', ':', 'restart']    
    subprocess.run(CMD_split)

def drop_partial(trim_list, size=target_second): # drop partial audio and silence audio
    count_partial = 0
    count_silence = 0
    for name in trim_list:
        real_name = os.path.join(Trimmed_PATH, name)
        if sox.file_info.duration(real_name) < size:
            count_partial += 1
            os.remove(real_name)
            print("finishing "+str(int(trim_list.index(name)/len(trim_list)*100))+'%')
        elif sox.file_info.silent(real_name, threshold=0.0001) == True:
            count_silence += 1
            os.remove(real_name)
            print("finishing "+str(int(trim_list.index(name)/len(trim_list)*100))+'%')
    print("finished, drop "+str(count_partial)+" partial audios and "+str(count_silence)+" silence audios")

if __name__ == '__main__':
    # set time size in ENV
    # always only work on the unified data for triming
    src_directory = Unified_PATH
    dest_directory = Trimmed_PATH
    
    # created the trimmed directory, if already exist, do nothing
    create_trimmed_directory(src_directory, dest_directory)
    
    # trim ch and we in batch
    trim_batch(src_directory, dest_directory, "ch")
    trim_batch(src_directory, dest_directory, "we")

    # get the trimmed list
    trimed_profile = Profiler(Trimmed_PATH)

    # drop the partial audio
    trimed_list_ch = trimed_profile.wav_list["ch"]
    drop_partial(trimed_list_ch)

    trimed_list_we = trimed_profile.wav_list["we"]
    drop_partial(trimed_list_we)