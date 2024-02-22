import os
from ENV import Unified_PATH, target_second, Trimmed_PATH, segment_method

import shutil
import subprocess
from dataprofile import Profiler
import sox
from pretrim import pre_trim

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

def pad_partial(trim_list, size=target_second):
    # still need to remove all the silence audio
    count_partial = 0
    count_silence = 0
    for name in trim_list:
        real_name = os.path.join(Trimmed_PATH, name)
        if sox.file_info.duration(real_name) < size:
            count_partial += 1
            padding(real_name)
            print("finishing "+str(int(trim_list.index(name)/len(trim_list)*100))+'%')
        elif sox.file_info.silent(real_name, threshold=0.0001) == True:
            count_silence += 1
            os.remove(real_name)
            print("finishing "+str(int(trim_list.index(name)/len(trim_list)*100))+'%')
    print("finished, pad "+str(count_partial)+" partial audios and remove "+str(count_silence)+" silence audios")

def padding(real_name, size=target_second): # pad the head of the audio to the end
    # this real_name is the name of the tail segment
    def find_first_segment(src):
        # src -> first_segment: the first segment of the current song
        # eg: src = trimmed_30_Padding/ch/37/wav00/wav00_007.wav (the last segment of the current song)
        # first_segment = trimmed_30_Padding/ch/37/wav00/wav00_001.wav (the first segment of the current song)
        src_list = src.split("/")
        song_name = src_list[-1].split("_")[0]
        song_folder = "/".join(src_list[:-1]) + "/" + song_name
        i = 1
        first_segment = song_folder + "_00"+str(i)+".wav"
        while not os.path.exists(first_segment):
            i += 1
            first_segment = song_folder + "_00"+str(i)+".wav"
        
        return first_segment
    first_segment = find_first_segment(real_name)
    # print(real_name) # the "tail" segment
    # print(first_segment) # the "head" segment
    # contact them together then trim the whole segment into "size"
    def circular_pad_single(tail, head):
        # pad the head of the audio to the end
        # then trim the whole segment into "size"
        
        # 4 steps:
        # CMD_pad = sox tail head temp.wav
        CMD_pad = ['sox', tail, head, 'temp.wav']
        subprocess.run(CMD_pad)
        # CMD_trim = sox sox temp.wav temp_trimed.wav trim 0 target_second
        CMD_trim = ['sox', 'temp.wav', 'temp_trimed.wav', 'trim', '0', str(target_second)]
        subprocess.run(CMD_trim)
        # CMD_rename = mv temp_trimed.wav tail
        CMD_rename = ['mv', 'temp_trimed.wav', tail]
        subprocess.run(CMD_rename)
        # CMD_remove = rm temp.wav
        CMD_remove = ['rm', 'temp.wav']
        subprocess.run(CMD_remove)
    
    def silence_pad_single(tail):
        # pad a silence audio to the end
        # then trim the whole segment into "size"
        # so the head is not needed
        
        # 5 steps:
        # create a target_second length silence audio
        # CMD_silence = sox -n -r 16000 silence.wav trim 0.0 target_second
        CMD_silence = ['sox', '-n', '-r', '16000', 'silence.wav', 'trim', '0.0', str(target_second)]
        subprocess.run(CMD_silence)
        # pad the silence audio to the end
        # CMD_pad = sox tail silence.wav temp.wav
        CMD_pad = ['sox', tail, 'silence.wav', 'temp.wav']
        subprocess.run(CMD_pad)
        # trim the whole segment into "size"
        # CMD_trim = sox temp.wav temp_trimed.wav trim 0 target_second
        CMD_trim = ['sox', 'temp.wav', 'temp_trimed.wav', 'trim', '0', str(target_second)]
        subprocess.run(CMD_trim)
        # rename the temp_trimed.wav to tail
        # CMD_rename = mv temp_trimed.wav tail
        CMD_rename = ['mv', 'temp_trimed.wav', tail]
        subprocess.run(CMD_rename)
        # remove the temp.wav and silence.wav
        # CMD_remove = rm temp.wav silence.wav
        CMD_remove = ['rm', 'temp.wav', 'silence.wav']
        subprocess.run(CMD_remove)

    if segment_method.endswith("C"):
        circular_pad_single(real_name, first_segment)
    if segment_method.endswith("S"):
        silence_pad_single(real_name)

if __name__ == '__main__':
    # set time size in ENV
    # always only work on the unified data for triming
    src_directory = Unified_PATH
    dest_directory = Trimmed_PATH
    
    '''pretrim block'''
    # pretrim the data
    print("pretrimming the data ... ... do not interrupt!")
    pre_trim(src_directory)
    print("pretrimming done")
    '''pretrim block'''

    '''trimming block'''
    # created the trimmed directory, if already exist, do nothing
    create_trimmed_directory(src_directory, dest_directory)
    
    # trim ch and we in batch
    trim_batch(src_directory, dest_directory, "ch")
    trim_batch(src_directory, dest_directory, "we")
    '''trimming block'''

    '''partial solution block'''
    # get the trimmed list
    trimed_profile = Profiler(Trimmed_PATH)

    if segment_method == "Dropping":
        # drop the partial audio
        trimed_list_ch = trimed_profile.wav_list["ch"]
        drop_partial(trimed_list_ch)

        trimed_list_we = trimed_profile.wav_list["we"]
        drop_partial(trimed_list_we)
    
    if segment_method.startswith("Padding"):
        trimed_list_ch = trimed_profile.wav_list["ch"]
        pad_partial(trimed_list_ch)

        trimed_list_we = trimed_profile.wav_list["we"]
        pad_partial(trimed_list_we)
    '''partial solution block'''