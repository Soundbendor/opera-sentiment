from dataset import SimpleAudioClassificationDataset
from xvalid_split import load_folds, print_folds, get_balance_folds, get_folds_to_size, find_biggest_diff, get_path_folds, get_song_id_path_folds
from record_gen import get_audio_name
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import copy

hyperparams = {}
hyperparams['input_size'] = 1024
hyperparams['batch_size'] = 32
hyperparams['output_size'] = 1

from ENV import Trimmed_PATH as mother_path

# fold related parameters
from ENV import target_second as piece_size
# usually no need to change this
from ENV import fold_count
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

def how_many_in_dataset(dataset):
    i = 0
    for _ in dataset.train.take(10000000):
        i+=1
    return i

# making data_full_dictionary
'''using load params'''
for root, dirs, files in os.walk(mother_path):
    for dir in dirs:
        if "wav" in dir: # trimmed_30/ch/9/wav00/
            # check if "in" folder under it is empty
            if os.path.exists(os.path.join(root,dir,"in")): # for the case that the whole recording is dropped (due to being shorter than trimming size)
                data_path = os.path.join(root,dir)
                # print("data_path", data_path)
                data_name = get_audio_name(data_path, mother_path)
                # print("data_name", data_name)
                dataset = SimpleAudioClassificationDataset(data_path, data_name)
                dataset._load_params(hyperparams['batch_size'],1.,0.,0.)
                raw = tf.data.TFRecordDataset(dataset._get_records())
                dataset.train = raw.map(lambda x: dataset._load_map(x, hyperparams["input_size"], dataset._length*dataset._sample_rate, hyperparams['output_size']))
                # print("the length of the dataset is: ", how_many_in_dataset(dataset))
                data_full_dictionary[data_path] = dataset
            else:
                pass

# print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav00"]))
# print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav01"]))
# print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav02"]))

'''
# data_full_dictionary example:
                # data_full_dictionary['trimmed_30_Padding/ch/20/wav00'] = the simple audio classification dataset
                # you can use .train to get the dataset
                # eg: data_full_dictionary['trimmed_30_Padding/ch/20/wav00'].train.take(n)
                # n can be at most of the number of pieces in that path
'''
# data_full_dictionary: {'mono/we/mal_08/pos_7': <*dataset BEFORE being batched*>, 'mono/we/mal_08/pos_8': ... ...

# SimpleAudioClassificationDataset: SACDataset
# right now we need to get: dataset_of_folds_dictionary: fold#: concatenated_dataset
dataset_of_folds_dictionary = {} # SACDataset
for fold_n, paths in path_folds.items():
    path_first = paths[0]
    name_first = get_audio_name(path_first, mother_path)
    # HAVE to reload the first one to prevent shallow copy
    dataset_of_folds_dictionary[fold_n] = SimpleAudioClassificationDataset(path_first, name_first)
    dataset_of_folds_dictionary[fold_n]._load_params(hyperparams['batch_size'],1.,0.,0.)
    raw = tf.data.TFRecordDataset(dataset_of_folds_dictionary[fold_n]._get_records())
    dataset_of_folds_dictionary[fold_n].train = raw.map(lambda x: dataset_of_folds_dictionary[fold_n]._load_map(x, hyperparams["input_size"], dataset_of_folds_dictionary[fold_n]._length*dataset_of_folds_dictionary[fold_n]._sample_rate, hyperparams['output_size']))
    for i in range(1, len(paths)):
        # print("***** ***** *****")
        # print(fold_n)
        # print(paths[i])
        # print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav00"]))
        # print("***** ***** *****")
        if paths[i] in data_full_dictionary: # for the case that the whole recording is dropped (due to being shorter than trimming size)
            dataset_curr = data_full_dictionary[paths[i]] # SACDataset
            combined_train_dataset = dataset_of_folds_dictionary[fold_n].train.concatenate(dataset_curr.train) # TFData 
            # use .train because we make it 100% train, the test and val are empty, we will manually xvalid later
            dataset_of_folds_dictionary[fold_n].train = combined_train_dataset # TFData

# dataset_of_folds_dictionary: {1~5: <*dataset BEFORE being batched*>}
            
# print(dataset_of_folds_dictionary)

# print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav00"]))
# print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav01"]))
# print(how_many_in_dataset(data_full_dictionary["trimmed_30_Padding/ch/20/wav02"]))

# print("first check: the length of the dataset in oroginal dictionary is: ", how_many_in_dataset(data_full_dictionary['trimmed_30_Padding/ch/20/wav00']))
'''
dataset_of_folds_song_level_dictionary will look like:
{fold#: {song_id: [dataset, dataset (pieces before being batched)], 
        song_id: [dataset, dataset,...]},
fold#: {song_id: [dataset, dataset,...},...}
'''
'''
maybe better to make it like:
{fold#: {song_id: {path: [dataset, dataset, ...], path: [dataset, dataset, ...], ...},
        song_id: {path: [dataset, dataset, ...], path: [dataset, dataset, ...], ...},
        ...},
fold#: {song_id: {path: [dataset, dataset, ...], path: [dataset, dataset, ...], ...}, ...}
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
            
            '''
            # # manually check size
            # dataset_size = how_many_in_dataset(data_full_dictionary[single_path])
            # folder_path = os.path.join(single_path, "in")
            # folder_size = 0
            # # check how many wav file under "folder_path"
            # for root, dirs, files in os.walk(folder_path):
            #     for file in files:
            #         if file.endswith(".wav"):
            #             folder_size+=1
            # wrong_flag = False
            # if dataset_size != folder_size:
            #     print("dataset size and folder size do not match: ", single_path)
            #     print("dataset size: ", dataset_size)
            #     print("folder size: ", folder_size)
            #     wrong_flag = True
            # if not wrong_flag:
            #     print(single_path, " dataset size GOOD!!!")
            '''


# testing
if __name__ == "__main__":
    # print(data_full_dictionary)
    # print(dataset_of_folds_dictionary)
    # dataset = dataset_of_folds_dictionary[1] # get fold_1
    # for batch in dataset.train.take(1):
    #     print(batch[0])
    #     print(batch[1])
    
    # print("***** ***** *****")
    # print(dataset_of_folds_song_level_dictionary)
    # print(dataset_of_folds_song_level_dictionary[1])
    # print(dataset_of_folds_song_level_dictionary[1][20])
    # print(dataset_of_folds_song_level_dictionary[1][20]['trimmed_30_Padding/ch/20/wav00'].train)
    
    # i = how_many_in_dataset(dataset_of_folds_song_level_dictionary[1][20]['trimmed_30_Padding/ch/20/wav00'])
    # print("there are ", i, " pieces of data in this dataset")
    print("Data loaded successfully!")