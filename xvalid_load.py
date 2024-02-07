from dataset import SimpleAudioClassificationDataset
from xvalid_split import load_folds, print_folds, get_balance_folds, get_folds_to_size, find_biggest_diff, get_path_folds
from record_gen import get_audio_name
import pandas as pd
import os
import tensorflow as tf
import numpy as np

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
load_name = "ch_folds_30_Padding_0.npy"
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

print(path_folds)

# right now we need to load all the data and concatenate dataset_of_folds_dictionary
data_full_dictionary = {}

# # '''using load no repeated''' # <----- still the old version
# # for path in df['file_path']:
# #     data_path = os.path.join(mother_folder, path)
# #     data_name = get_audio_name(data_path, mother_folder)
# #     dataset = SimpleAudioClassificationDataset(data_path, data_name)
# #     dataset.load_no_repeat(hyperparams['input_size'], hyperparams['batch_size'], hyperparams['output_size'], train_split = 1, val_split=0, test_split=0)
# #     data_full_dictionary[path] = dataset

# making data_full_dictionary
'''using load params'''
for root, dirs, files in os.walk(mother_path):
    for dir in dirs:
        if "wav" in dir: # trimmed_30/ch/9/wav00/
            # check if "in" folder under it is empty
            if os.path.exists(os.path.join(root,dir,"in")): # for the case that the whole recording is dropped (due to being shorter than trimming size)
                data_path = os.path.join(root,dir)
                data_name = get_audio_name(data_path, mother_path)
                dataset = SimpleAudioClassificationDataset(data_path, data_name)
                dataset._load_params(hyperparams['batch_size'],1.,0.,0.)
                raw = tf.data.TFRecordDataset(dataset._get_records())
                dataset.train = raw.map(lambda x: dataset._load_map(x, hyperparams["input_size"], dataset._length*dataset._sample_rate, hyperparams['output_size']))
                data_full_dictionary[data_path] = dataset

# # data_full_dictionary: {'mono/we/mal_08/pos_7': <*dataset BEFORE being batched*>, 'mono/we/mal_08/pos_8': ... ...
# SimpleAudioClassificationDataset: SACDataset
# right now we need to get: dataset_of_folds_dictionary: fold#: concatenated_dataset
dataset_of_folds_dictionary = {} # SACDataset
for fold_n, paths in path_folds.items():
    dataset_first = data_full_dictionary[paths[0]] # SACDataset
    dataset_of_folds_dictionary[fold_n] = dataset_first # SACDataset
    for i in range(1, len(paths)):
        if paths[i] in data_full_dictionary: # for the case that the whole recording is dropped (due to being shorter than trimming size)
            dataset_curr = data_full_dictionary[paths[i]] # SACDataset
            combined_train_dataset = dataset_of_folds_dictionary[fold_n].train.concatenate(dataset_curr.train) # TFData 
            # use .train because we make it 100% train, the test and val are empty, we will manually xvalid later
            dataset_of_folds_dictionary[fold_n].train = combined_train_dataset # TFData

# dataset_of_folds_dictionary: {1~5: <*dataset BEFORE being batched*>}
print(dataset_of_folds_dictionary)

dataset_of_folds_song_level_dictionary = {} # SACDataset
'''
this variable will look like:


'''

# testing
# if __name__ == "__main__":
    # dataset = dataset_of_folds_dictionary[1] # get fold_1
    # for batch in dataset.train.take(100):
    #     print(batch[0])
    #     print(batch[1])