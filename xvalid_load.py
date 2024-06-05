#Pytorch
from xvalid_split import load_folds, print_folds, get_balance_folds, get_folds_to_size, find_biggest_diff, get_path_folds, get_song_id_path_folds
from csv_gen import get_audio_name
import pandas as pd
import os
import numpy as np
import copy
from Opera2023Dataset import Opera2023Dataset, Opera2023Dataset_Spec, Opera2023DatasetMelody, Opera2023Dataset_lyrics_bert
from HYPERPARAMS import hyperparams
from torch.utils.data import ConcatDataset


from ENV import Trimmed_PATH as mother_path

# fold related parameters
from ENV import target_second as piece_size
# 
from ENV import REPRESENTATION
# usually no need to change this
from ENV import fold_count, target_class
lan = "ch"

# load a folds shape, remember to change the end number if you want to load a different fold, 
# default will be the first one: 1
load_name = "ch_folds_30_Padding-S_0.npy"
folds = load_folds(load_name)

# if you want to generate a new folds
# folds = get_balance_folds(fold_count, lan)

print("the folds we are using are:")
print_folds(folds)
print("the size of each fold are:")
folds_size = get_folds_to_size(folds, lan)
print(folds_size)
print("***** ***** *****")
print("the representation we are using is: ", REPRESENTATION)
''' explaination for path_fold, folds_pattern, dataset_of_folds_dictionary
    path_folds: fold#: file_path
    {1:[path_list], 2: [path_list], 3:[path_list]. 4:[path_list]}

    dataset_of_folds_dictionary: {dataset_X: all the dataset in fold X}

    the folds_all will look like:
    folds_pattern = {0: [[2, 3, 4, 5], [1]], 1: [[1, 3, 4, 5], [2]], 2: [[1, 2, 4, 5], [3]],
                3: [[1, 2, 3, 5], [4]], 4: [[1, 2, 3, 4], [5]]}
    
# '''
#get path_folds
path_folds = get_path_folds(mother_path, lan, folds) # convert folds to song_id into folds to file_path

# print(path_folds)

# get song_id_path_folds
song_id_path_folds = get_song_id_path_folds(mother_path, lan, folds) # convert folds to song_id into folds to file_path
# print(song_id_path_folds)

# right now we need to load all the data and concatenate dataset_of_folds_dictionary
data_full_dictionary = {}

# making data_full_dictionary
for root, dirs, files in os.walk(mother_path):
    for dir in dirs:
        if "wav" in dir: # trimmed_30/ch/9/wav00/
            # check if "in" folder under it is empty
            if os.path.exists(os.path.join(root,dir,"in")): # for the case that the whole recording is dropped (due to being shorter than trimming size)
                data_dir = os.path.join(root,dir)
                csv_name = get_audio_name(data_dir, mother_path)+".csv"
                csv_dir = os.path.join(data_dir, csv_name)
                if REPRESENTATION == "raw":
                    # generate raw waveform
                    dataset = Opera2023Dataset(csv_dir, data_dir, target_class, hyperparams["input_size"])
                elif REPRESENTATION in ["mel", "mfcc"]:
                    # generate mel spectrogram or mfcc
                    dataset = Opera2023Dataset_Spec(csv_dir, data_dir, target_class, REPRESENTATION)
                elif REPRESENTATION == "melody":
                    dataset = Opera2023DatasetMelody(csv_dir, data_dir, target_class, hyperparams["input_size"])
                elif REPRESENTATION == "lyrics":
                    dataset = Opera2023Dataset_lyrics_bert(csv_dir, data_dir, target_class)
                else:
                    raise ValueError("REPRESENTATION not supported")
                data_full_dictionary[data_dir] = dataset
            else:
                pass

'''
data_full_dictionary example:
data_full_dictionary['trimmed_30_Padding/ch/20/wav00'] = A Opera2023Dataset dataset
'''

# right now we need to get: dataset_of_folds_dictionary: fold#: concatenated_dataset
dataset_of_folds_dictionary = {} # SACDataset
for fold_n, paths in path_folds.items():
    # maintain a concat list
    concatenate_later_list = []
    for i in range(len(paths)):
        if paths[i] in data_full_dictionary: # for the case that the whole recording is dropped (due to being shorter than trimming size)
            dataset = data_full_dictionary[paths[i]]
            concatenate_later_list.append(dataset)
    
    dataset_of_folds_dictionary[fold_n] = ConcatDataset(concatenate_later_list) # concatenate now

# dataset_of_folds_dictionary: {1~5: <torch.utils.data.dataset.ConcatDataset object at 0x7f519a7a16f0> (before batch)}

'''
dataset_of_folds_song_level_dictionary will look like:
{fold#: {song_id: [dataset, dataset (each dataset is a recording)], 
        song_id: [dataset, dataset,...]},
fold#: {song_id: [dataset, dataset,...},...}

The inner dictionary level with song_id as kay is the song level
The inner list level is the recording level
'''
dataset_of_folds_song_level_dictionary = {} # for evaluation one by one

for fold_id, folds_distri in folds.items():
    dataset_of_folds_song_level_dictionary[fold_id] = {}
    for song_id in folds_distri:
        dataset_of_folds_song_level_dictionary[fold_id][song_id] = []
        # dataset_of_folds_song_level_dictionary[fold_id][song_id] = {}
        current_song_id_path = song_id_path_folds[fold_id][song_id]
        for single_path in current_song_id_path:
            dataset_of_folds_song_level_dictionary[fold_id][song_id].append(data_full_dictionary[single_path])
            # dataset_of_folds_song_level_dictionary[fold_id][song_id][single_path] = data_full_dictionary[single_path]

if __name__ == "__main__":
    print(data_full_dictionary)
    print(dataset_of_folds_dictionary)
    dataset = dataset_of_folds_dictionary[1] # get fold_1
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][0].shape)

    print("Data loaded successfully!")