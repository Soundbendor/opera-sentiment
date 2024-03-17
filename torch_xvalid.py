import numpy as np
import math
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from torchinfo import summary

from torchviz import make_dot
from torchview import draw_graph

from training_time import train
from Evaluator import Evaluator

from sklearn.metrics import confusion_matrix

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
hyperparams['epochs'] = 2
from ENV import segment_method

from ENV import NEPTUNE_SWITCH, Evaluate_Frequency
TEST_ON = 0 # 0 means using cross validation, 1-5 means the only fold to test on

MODEL = "DummyModel"
print("MODEL: ", MODEL)
method = "None"
print("method: ", method)

if NEPTUNE_SWITCH == 1:
    import neptune
    from neptune_pytorch import NeptuneLogger
    from neptune.utils import stringify_unsupported
    from neptune.types import File

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

    run["hyperparams"] = stringify_unsupported(hyperparams)
    run["sys/tags"].add([str(hyperparams['epochs'])+"epochs", MODEL, str(method)])
    run["sys/tags"].add(str(piece_size)+"s_"+segment_method)
    run["sys/tags"].add(str(hyperparams["batch_size"])+"batch_size")
    run["sys/tags"].add(target_class_dictionary[target_class])

    run["info/size of folds"] = folds_size
    run["info/model method"] = method
    run["target class"] = target_class_dictionary[target_class]


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

# round number to 2 digits
def round2(num):
    return round(num,2)

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

def get_checkpoints_list(name, epoch = hyperparams['epochs'], period = Evaluate_Frequency):
    if epoch < period:
        print("WARNING: epoch is less than period, no checkpoints will be generated")
        return []
    epoch_list = [epoch for epoch in range(1, epoch + 1) if epoch % period == 0]
    # Add the last epoch if it's not already included
    if epoch % period != 0:
        epoch_list.append(epoch)
    output = []
    for num in epoch_list:
        output.append(name + "_" + str(num) + ".pt")
    return output

def get_metrics(confusion_matrix):
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]

    precision = round2(TP/(TP+FP))
    recall = round2(TP/(TP+FN))
    F1 = round2(2 * (precision * recall) / (precision + recall))
    return precision, recall, F1

# save confusion matrix as a image file
def save_conf_matrix(conf_matrix, file_name):
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])
    fig = ax.get_figure()
    fig.savefig(file_name)
    plt.clf()
    if NEPTUNE_SWITCH == 1:
        neptune_name = file_name.split("/")[-1]
        run[neptune_name].upload(File(file_name))

def save_normed_conf_matrix(cm, file_name):
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cmn.flatten().astype(float)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cmn, annot=labels, fmt='', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    ax.set_title('Normalized Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    fig = ax.get_figure()
    fig.savefig(file_name)
    plt.clf()
    if NEPTUNE_SWITCH == 1:
        neptune_name = file_name.split("/")[-1]
        run[neptune_name].upload(File(file_name))

def my_x_validation(dataset_of_folds_dictionary, model_class, device, fold_count, test_on = 0):
    folds_pattern = get_folds_pattern(fold_count)

    model = model_class("Dropout03").to(device)
    input_size = (1, math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])
    model_summary = summary(model, input_size, device=device)
    
    dummy_data = torch.randn(input_size).to(device)
    
    y = model(dummy_data)
    image_name_tv = "model_torchviz"
    image_formate = "png"
    image_path_tv = image_name_tv + "." + image_formate
    make_dot(y, params=dict(model.named_parameters())).render(image_name_tv, format=image_formate)

    image_name_torchview = "model_torchview"
    image_path_torchview = image_name_torchview + "." + image_formate
    draw_graph(model = model, input_size = input_size, device=device, save_graph=True, filename=image_name_torchview, roll=False)

    if NEPTUNE_SWITCH == 1:
        npt_logger = NeptuneLogger(
        run, model=model, log_model_diagram=True, log_gradients=True, log_parameters=True
        )
        run["model_summary"] = str(model_summary)
        run[image_path_tv].upload(File(image_path_tv))
        run[image_path_torchview].upload(File(image_path_torchview))

    # if test on is on one fold, then only test on that fold
    if test_on in range(1,fold_count+1):
        folds_pattern = {test_on-1: folds_pattern[test_on-1]}

    targets_full_seg = []
    targets_full_rec = []
    targets_full_song = []
    predictions_full_seg = []
    predictions_full_rec = []
    predictions_full_song = []

    # each loop is one fold validation
    for i, (train_index, test_index) in folds_pattern.items():
        # every fold needs to have a new model
        model = model_class("Dropout03").to(device)
        
        # get training set
        dataset_train_list = []
        for j in range(len(train_index)):
            dataset_train_list.append(dataset_of_folds_dictionary[train_index[j]])
        dataset_train = ConcatDataset(dataset_train_list)
        # get test set
        dataset_test = dataset_of_folds_dictionary[test_index[0]]

        train_loader = DataLoader(dataset_train, batch_size=hyperparams['batch_size'], shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=hyperparams['batch_size'], shuffle=True)

        if hyperparams["loss"] == 'BCE':
            loss_function = nn.BCELoss()
        else:
            raise ValueError("loss functio not supported")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])


        # ====== prepare model folder ======
        # remove everything in 'models/MODEL-method' folder if it exists, otherwise create a empty one
        models_folder = "models/"+MODEL+"_"+method+"/" # eg: models/LSTM_drop0.3/
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        else:
            filelist = [ f for f in os.listdir(models_folder) ]
            for f in filelist:
                os.remove(os.path.join(models_folder, f))
        
        checkpoint_name = models_folder+'model_weights' # eg: models/LSTM_drop0.3/model_weights

        train(model, train_loader, loss_function, optimizer, device, hyperparams['epochs'], run, npt_logger, 
              checkpoint_name = checkpoint_name, 
              evaluate_frequency = Evaluate_Frequency)
        
        checkpoints_list = get_checkpoints_list(checkpoint_name, hyperparams['epochs'], Evaluate_Frequency)

        # evaluate on each checkpoint
        for checkpoint in checkpoints_list[0:-1]:
            print("evaluating on checkpoint:", checkpoint)
            model.load_state_dict(torch.load(checkpoint))
            evaluator_epochly = Evaluator(model, loss_function, device, run, npt_logger)
            acc_seg = evaluator_epochly.evaluate_segment(test_loader)
            acc_rec, acc_song = evaluator_epochly.evaluate_recording_and_song(dataset_of_folds_song_level_dictionary[test_index[0]])
            
        # load back the final model
        final_checkpoint = checkpoints_list[-1]
        model.load_state_dict(torch.load(final_checkpoint))

        print("****** final validating on fold", test_index,"******")

        evaluator = Evaluator(model, loss_function, device, run, npt_logger)
        acc_seg = evaluator.evaluate_segment(test_loader)
        acc_rec, acc_song = evaluator.evaluate_recording_and_song(dataset_of_folds_song_level_dictionary[test_index[0]])

        print("fold", i, "segment accuracy:", acc_seg, "recording accuracy:", acc_rec, "song accuracy:", acc_song)

        targets_full_seg.extend(evaluator.targets_seg)
        targets_full_rec.extend(evaluator.targets_rec)
        targets_full_song.extend(evaluator.targets_song)
        predictions_full_seg.extend(evaluator.predictions_seg)
        predictions_full_rec.extend(evaluator.predictions_rec)
        predictions_full_song.extend(evaluator.predictions_song)

        # reset model
        model = model_class("Dropout03").to(device)
    
    # calculate the aggregate average accuracy
    acc_seg_avg = sum([1 if targets_full_seg[i] == predictions_full_seg[i] else 0 for i in range(len(targets_full_seg))]) / len(targets_full_seg)
    acc_rec_avg = sum([1 if targets_full_rec[i] == predictions_full_rec[i] else 0 for i in range(len(targets_full_rec))]) / len(targets_full_rec)
    acc_song_avg = sum([1 if targets_full_song[i] == predictions_full_song[i] else 0 for i in range(len(targets_full_song))]) / len(targets_full_song)
    
    acc_seg_avg = round2(acc_seg_avg)
    acc_rec_avg = round2(acc_rec_avg)
    acc_song_avg = round2(acc_song_avg)

    print("average segment accuracy:", acc_seg_avg, "average recording accuracy:", acc_rec_avg, "average song accuracy:", acc_song_avg)
    
    conf_matrix_seg = confusion_matrix(targets_full_seg, predictions_full_seg)
    conf_matrix_rec = confusion_matrix(targets_full_rec, predictions_full_rec)
    conf_matrix_song = confusion_matrix(targets_full_song, predictions_full_song)

    precision_segment, recall_segment, F1_segment = get_metrics(conf_matrix_seg)
    precision_recording, recall_recording, F1_recording = get_metrics(conf_matrix_rec)
    precision_song, recall_song, F1_song = get_metrics(conf_matrix_song)

    # save confusion matrix
    save_conf_matrix(conf_matrix_seg, './cfMatrix_seg.png')
    save_normed_conf_matrix(conf_matrix_seg, './cfMatrix_seg_normed.png')
    save_conf_matrix(conf_matrix_rec, './cfMatrix_recor.png')
    save_normed_conf_matrix(conf_matrix_rec, './cfMatrix_recor_normed.png')
    save_conf_matrix(conf_matrix_song, './cfMatrix_song.png')
    save_normed_conf_matrix(conf_matrix_song, './cfMatrix_song_normed.png')

    if NEPTUNE_SWITCH == 1:
        run["result/seg/acc"] = acc_seg_avg
        run["result/seg/precision"] = precision_segment
        run["result/seg/recall"] = recall_segment
        run["result/seg/F1"] = F1_segment

        run["result/rec/acc"] = acc_rec_avg
        run["result/rec/precision"] = precision_recording
        run["result/rec/recall"] = recall_recording
        run["result/rec/F1"] = F1_recording

        run["result/song/acc"] = acc_song_avg
        run["result/song/precision"] = precision_song
        run["result/song/recall"] = recall_song
        run["result/song/F1"] = F1_song

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
    model_class = DummyModel

    my_x_validation(dataset_of_folds_dictionary, model_class, device, fold_count, TEST_ON)
                                                                # 0 means using cross validation
                                                                # change this into the only test fold you want to use
    
    if NEPTUNE_SWITCH == 1:
        run.stop()