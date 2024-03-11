import numpy as np
import math
import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from training_time import train
from Evaluator import Evaluator

from xvalid_load import folds, folds_size, data_full_dictionary, dataset_of_folds_dictionary, dataset_of_folds_song_level_dictionary

from ENV import Trimmed_PATH as path # dataset path
from ENV import target_second as piece_size

# HOW to get folds? go to "xvalid_load.py" to load or create new ones

from ENV import fold_count
from ENV import target_class, target_class_dictionary

from HYPERPARAMS import hyperparams
import sys
# MODEL = sys.argv[1]
# method = sys.argv[2]

# hyperparams['epochs'] = int(sys.argv[3])
hyperparams['epochs'] = 1
from ENV import segment_method

from ENV import NEPTUNE_SWITCH, Evaluate_Frequency
TEST_ON = 0 # 0 means using cross validation, 1-5 means the only fold to test on

# print("MODEL: ", MODEL)
print("epochs: ", hyperparams['epochs'])
# print("method: ", method)

if NEPTUNE_SWITCH == 1:
    import neptune
    from neptune_pytorch import NeptuneLogger
    from neptune.utils import stringify_unsupported

    import configparser
    def _process_api_key(f_key: str) -> configparser.ConfigParser:
        api_key = configparser.ConfigParser()
        api_key.read(f_key)
        return api_key
    
    creds = _process_api_key('./neptune.ini')

    run = neptune.init_run(
        project=creds['CLIENT_INFO']['project_id'],
        api_token=creds['CLIENT_INFO']['api_token']
    )

''' folds_pattern example: train on ... test on ...
{0: [[2, 3, 4, 5], [1]], 
 1: [[1, 3, 4, 5], [2]], 
 2: [[1, 2, 4, 5], [3]], 
 3: [[1, 2, 3, 5], [4]], 
 4: [[1, 2, 3, 4], [5]]}
'''
def get_folds_pattern(fold_count):
    folds_pattern = {}
    for i in range(fold_count):
        folds_pattern[i] = [[j for j in range(1, fold_count + 1) if j != i + 1], [i + 1]]
        
    return folds_pattern

# covert list[numpy.ndarray] into pure one dimensional list
def super_flat(arrays):
    if type(arrays[0]) != np.ndarray:
        return arrays
    flat_array = np.concatenate(arrays, axis=0)
    flat_list = flat_array.tolist()
    # flat the flat_list
    flat_list = [item for sublist in flat_list for item in sublist]
    return flat_list

def flatten_nested_list(nested_list):
    temp_list = []
    for sublist in nested_list:
        temp_list.extend(sublist)
    return temp_list

def my_x_validation(dataset_of_folds_dictionary, model_class, device, fold_count, test_on = 0):
    folds_pattern = get_folds_pattern(fold_count)

    model = model_class("Dropout03").to(device)

    if NEPTUNE_SWITCH == 1:
        npt_logger = NeptuneLogger(
        run, model=model, log_model_diagram=True, log_gradients=True, log_parameters=True, log_freq=30
        )
        run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(hyperparams)

    # if test on is on one fold, then only test on that fold
    if test_on in range(1,fold_count+1):
        folds_pattern = {test_on-1: folds_pattern[test_on-1]}

    # each loop is one fold validation
    for i, (train_index, test_index) in folds_pattern.items():
        # every fold needs to have a new model
        model = model_class("Dropout03").to(device)
        
        dataset_train_list = []
        for j in range(len(train_index)):
            dataset_train_list.append(dataset_of_folds_dictionary[train_index[j]])
        dataset_train = ConcatDataset(dataset_train_list)
        dataset_test = dataset_of_folds_dictionary[test_index[0]]

        train_loader = DataLoader(dataset_train, batch_size=hyperparams['batch_size'], shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=hyperparams['batch_size'], shuffle=True)

        if hyperparams["loss"] == 'BCE':
            loss_function = nn.BCELoss()
        else:
            raise ValueError("loss functio not supported")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])

        train(model, train_loader, loss_function, optimizer, device, hyperparams['epochs'], run, npt_logger)

        evaluate(model, test_loader, loss_function, device, run, npt_logger)

        # reset model
        model = DummyModel().to(device)

def print_running_information():
    print("running information:")
    print("running on data:", path)
    print("***** ***** *****")
    print("hyperparams:")
    for i in hyperparams:
        print(i, hyperparams[i])
    print("***** ***** *****")

if __name__ == "__main__":
    print_running_information()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    from Models import DummyModel, LSTM
    model_class = LSTM

    my_x_validation(dataset_of_folds_dictionary, model_class, device, fold_count, TEST_ON)
                                                                # 0 means using cross validation
                                                                # change this into the only test fold you want to use
    
    if NEPTUNE_SWITCH == 1:
        run.stop()