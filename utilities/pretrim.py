# trimming down the silence parts from the begining and end of the audio files
# do this before or after unifying, but before triming!!!

import sox
import os
import subprocess

def single_pre_trim(src, dest):
    # sox in.wav out.wav silence 1 0.1 0.2% reverse silence 1 0.1 0.2% reverse
    CMD_trim_pre = ['sox', src, dest, 'silence', '1', '0.1', '0.2%', 'reverse', 'silence', '1', '0.1', '0.2%', 'reverse']
    subprocess.run(CMD_trim_pre)
    CMD_rename = ['mv', dest, src]  # rename temp.wav to original name (overwrite)
    subprocess.run(CMD_rename)

def pre_trim(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".wav"):
                src = os.path.join(root, file)
                dest = os.path.join(root, 'temp.wav')
                single_pre_trim(src, dest)

if __name__ == '__main__':
    src_dir = "/nfs/guille/eecs_research/soundbendor/shengxuan/opera2324/opera-singing-dataset/operadataset2023"
    pre_trim(src_dir)