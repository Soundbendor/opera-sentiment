import os
import subprocess
import numpy as np

# -> all the .wav files in the path, type: list
# unify the extension to lower case at the same time
def get_wavlist(path):
    wavelist = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.join(path, file_name).endswith('.wav'):
                wavelist.append(os.path.join(path, file_name))
            if os.path.join(path, file_name).endswith('.WAV'):
                new_name = file_name.replace('WAV', 'wav')
                os.rename(os.path.join(path, file_name), os.path.join(path, new_name))
                wavelist.append(os.path.join(path, new_name))
    return wavelist

# -> duration of in second, type: float
def get_duration(name): 
    CMD = ['soxi', '-D', name] # float
    duration = subprocess.run(CMD, stdout=subprocess.PIPE)
    # transfer it into readable output
    dura_readable = duration.stdout.decode("utf-8")
    # float it
    dura_readable = float(dura_readable)
    return dura_readable

# -> channel amount, type: int
def get_channel(name): 
    CMD = ['soxi', '-c', name] # int
    channel = subprocess.run(CMD, stdout=subprocess.PIPE)
    # transfer it into readable output
    chan_readable = channel.stdout.decode("utf-8")
    # float it
    chan_readable = int(chan_readable)
    return chan_readable

# -> sample rate, type: int
def get_SR(name):
    CMD = ['soxi', '-r', name] # int
    SR = subprocess.run(CMD, stdout=subprocess.PIPE)
    # transfer it into readable output
    SR_readable = SR.stdout.decode("utf-8")
    # float it
    SR_readable = int(SR_readable)
    return SR_readable

# -> bit rate, type: int
def get_bit(name):
    CMD = ['soxi', '-b', name] # int
    bit = subprocess.run(CMD, stdout=subprocess.PIPE)
    # transfer it into readable output
    bit_readable = bit.stdout.decode("utf-8")
    # float it
    bit_readable = int(bit_readable)
    return bit_readable

# sorted dict by values
def sort_dict(dict):
    tuple = zip(dict.values(), dict.keys())
    rank = sorted(tuple)
    dict_sort = {}
    for i in range(len(rank)):
        dict_sort[rank[i][1]]=rank[i][0]