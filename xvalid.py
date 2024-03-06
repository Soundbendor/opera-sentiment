import time
import numpy as np
import math
import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import random

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

from sklearn.metrics import confusion_matrix, accuracy_score
from xvalid_load import folds, folds_size, data_full_dictionary, dataset_of_folds_dictionary, dataset_of_folds_song_level_dictionary, hyperparams

from dataset import SimpleAudioClassificationDataset

from ENV import Trimmed_PATH as path # dataset path
from ENV import target_second as piece_size

# HOW to get folds? go to "xvalid_load.py" to load or create new ones

from ENV import fold_count
from ENV import target_class, target_class_dictionary

'''Don't Change Here, if change needed, go back to xvalid_load.py to change these
hyperparams['input_size'] = 1024
hyperparams['batch_size'] = 32
hyperparams['output_size'] = 1
'''

# hyperparams['activation'] = "softmax"
hyperparams['dense_units'] = hyperparams['output_size']
hyperparams['dropout'] = 0.3
hyperparams['lr'] = 0.001
hyperparams["loss"] = 'binary_crossentropy' #'categorical_crossentropy'
hyperparams['input_length'] = 16000*piece_size

import sys
MODEL = sys.argv[1]
method = sys.argv[2]
hyperparams['epochs'] = int(sys.argv[3])
from ENV import segment_method


# # if manually set
# MODEL = "Bi_LSTM"
# method = 'shaun_adapted'
# hyperparams['epochs'] = 5


from ENV import NEPTUNE_SWITCH, Evaluate_Frequency
TEST_ON = 0 # 0 means using cross validation, 1-5 means the only fold to test on

print("MODEL: ", MODEL)
print("epochs: ", hyperparams['epochs'])
print("method: ", method)

if NEPTUNE_SWITCH == 1:
    ###### for Neptune ######
    import configparser
    import neptune
    from neptune.integrations.tensorflow_keras import NeptuneCallback
    from neptune.types import File

    def _process_api_key(f_key: str) -> configparser.ConfigParser:
        api_key = configparser.ConfigParser()
        api_key.read(f_key)
        return api_key

    def init_neptune(cfg: str):
        # You will need to store your neptune project id and api key
        # in an external file. Please do not hard-code these values - 
        # it is a security risk.
        # Do not commit this credentials file to your github repository.
        creds = _process_api_key(cfg)
        runtime = neptune.init_run(project=creds['CLIENT_INFO']['project_id'],
                            api_token=creds['CLIENT_INFO']['api_token'])
        return runtime, NeptuneCallback(run=runtime, base_namespace='metrics')
    
    runtime, neptune_cbk = init_neptune('./neptune.ini')
    # upload parameters to neptune
    runtime['parameters'] = hyperparams
    
    runtime["sys/tags"].add([str(hyperparams['epochs'])+"epochs", MODEL, str(method)])
    runtime["sys/tags"].add(str(piece_size)+"s_"+segment_method)
    runtime["sys/tags"].add(str(hyperparams["batch_size"])+"batch_size")
    runtime["sys/tags"].add(target_class_dictionary[target_class])
    
    runtime["info/size of folds"] = folds_size
    runtime["info/model method"] = method
    runtime["target class"] = target_class_dictionary[target_class]
    ###### for Neptune end ######

# Define a custom callback to save model visualization
class ModelVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(ModelVisualizationCallback, self).__init__()
        self.log_dir = log_dir
        self.generated_visualization = False

    def on_train_begin(self, logs=None):
        if not self.generated_visualization:
            model = self.model  # Get the model
            to_file=os.path.join(self.log_dir, "model.png")
            tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True)
            if NEPTUNE_SWITCH == 1:
                runtime["module_vis"].upload(File(to_file))
            self.generated_visualization = True

# Set up TensorBoard
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# round number to 2 digits
def round2(num):
    return round(num,2)

''' folds_pattern example
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


def shuffle_and_batch(dataset):
    return dataset.shuffle(buffer_size=1024).batch(hyperparams["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

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
        output.append(name + "_" + str(num) + ".h5")
    return output

# save confusion matrix as a file
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
        runtime[neptune_name].upload(File(file_name))

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
        runtime[neptune_name].upload(File(file_name))


'''input explaination:
dataset_of_folds_dictionary: {1~5: <*dataset BEFORE being batched*>}
folds_pattern: 
    {0: [[2, 3, 4, 5], [1]], 
     1: [[1, 3, 4, 5], [2]], 
     2: [[1, 2, 4, 5], [3]], 
     3: [[1, 2, 3, 5], [4]], 
     4: [[1, 2, 3, 4], [5]]}
'''
def my_x_validation(dataset_of_folds_dictionary,folds_pattern, test_on = 0):
    from evaluator import Evaluator_Segment, Evaluator_Recording_Song
    overall_random_pick_times = 0
    overall_voting_times = 0
    scores_segment_list = []
    scores_recording_list = []
    scores_song_list = []
    y_true_all_segment = []
    y_pred_all_segment = []
    y_true_all_recording = []
    y_pred_all_recording = []
    y_true_all_song = []
    y_pred_all_song = []

    score_segment_validation_epoch = []
    score_recording_validation_epoch = []
    score_song_validation_epoch = []

    # each loop is one fold validation
    if test_on in range(1,fold_count+1):
        folds_pattern = {test_on-1: folds_pattern[test_on-1]}
    
    for i, (train_index, test_index) in folds_pattern.items():
        # merge the training data:
        dataset_train = dataset_of_folds_dictionary[train_index[0]].train
        
        for j in range(1,len(train_index)):
            data_index_curr = train_index[j]
            dataset_train = dataset_train.concatenate(dataset_of_folds_dictionary[data_index_curr].train)
        dataset_test = dataset_of_folds_dictionary[test_index[0]].train
        # print("after concatenate:")
        # print(dataset_train)
        
        # shuffle and batch train and test data
        dataset_train = shuffle_and_batch(dataset_train)
        dataset_test = shuffle_and_batch(dataset_test)
        # print("after batch and shuffle:")
        # print(dataset_train)
        # print(dataset_test)
        
        # #####TESTING: check batched data size
        # for batch_x, batch_y in dataset_train:
        #     print("Batch x shape:", batch_x.shape)
        #     print("Batch y shape:", batch_y.shape)
        #     break

        # #####TESTING: check batched data size

        # create a new model for each fold
        model = tf.keras.Sequential()
        optim = model_adding(model)
        # model compile and build
        model.compile(optimizer=optim, loss=hyperparams["loss"], metrics=["accuracy"])
        model.build((hyperparams['batch_size'], None, hyperparams['input_size']))

        # since all the models are the same, we only need to print the summary once
        if i == 0: 
            model.summary()

        # remove everything in 'models/' folder if it exists, otherwise create a empty one
        models_folder = "models/"+MODEL+"_"+method+"/" # eg: models/LSTM_drop0.3/
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        else:
            filelist = [ f for f in os.listdir(models_folder) ]
            for f in filelist:
                os.remove(os.path.join(models_folder, f))

        checkpoint_name = models_folder+'model_weights' # eg: models/LSTM_drop0.3/model_weights
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_name+"_{epoch}.h5", 
                                                                save_weights_only=True, 
                                                                save_best_only=False,
                                                                verbose=1,
                                                                period=Evaluate_Frequency)  # Save every 5 epochs

        print("****** training on folds partern", i,"******")
        if NEPTUNE_SWITCH ==1:
          model.fit(dataset_train, epochs = hyperparams['epochs'], callbacks=[neptune_cbk, 
                                                                              tensorboard_callback, ModelVisualizationCallback(log_dir), 
                                                                              checkpoint_callback])
        else:
          model.fit(dataset_train, epochs = hyperparams['epochs'])
        

        ## evaluate on each __Evaluate_Frequency__ epochs besides the last one/final one
        checkpoints_list = get_checkpoints_list(checkpoint_name)
        for checkpoint in checkpoints_list[0:-1]:
            # checkpoint eg: 'models/model_weights_5.h5'
            model.load_weights(checkpoint)
            # this model becomes the model at this checkpoint, now we need to evaluate it using my own evaluator
            # later we will use this to replace the build in validation accuracy plot
            evaluator_segment = Evaluator_Segment(model, dataset_test, neptune_cbk)
            score_segment = evaluator_segment.get_score()
            evalauator_recording_song = Evaluator_Recording_Song(model, dataset_of_folds_song_level_dictionary[test_index[0]], neptune_cbk)
            score_recording, score_song = evalauator_recording_song.get_score()
            score_segment_validation_epoch.append(score_segment)
            score_recording_validation_epoch.append(score_recording)
            score_song_validation_epoch.append(score_song)
        
        # load back the final model
        final_checkpoint = checkpoints_list[-1]
        model.load_weights(final_checkpoint)

        print("****** final validating on fold", test_index,"******")
        
        ############################# ############################# ############################# #############################

        # segment evaluation
        evaluator_segment = Evaluator_Segment(model, dataset_test, neptune_cbk)
        score_segment = evaluator_segment.get_score()
        
        scores_segment_list.append(score_segment)
        score_segment_validation_epoch.append(score_segment) # append the final score to the epoch list

        y_true_fold_segment, y_pred_fold_segment = evaluator_segment.get_Y_labels()
        if len(y_true_fold_segment) != len(y_pred_fold_segment):
            print("y_true and y_pred are not the same length")
            print(len(y_true_fold_segment))
            print(len(y_pred_fold_segment))
            raise ValueError("y_true and y_pred are not the same length")
        
        # extend the true and predicted labels for the current fold to the overall lists
        y_true_all_segment.extend(y_true_fold_segment)
        y_pred_all_segment.extend(y_pred_fold_segment)
        # print(y_true_all_segment)
        # print(y_pred_all_segment)
        # print("******* segment level above ***************")
        # print("******* song level below ***************")
        
        ############################# ############################# ############################# #############################

        # song evaluation and recording evaluation (voting in different level)
        test_fold = test_index[0]
        evaluator_recording_song = Evaluator_Recording_Song(model, dataset_of_folds_song_level_dictionary[test_fold], neptune_cbk)

        recording_Y, song_Y = evaluator_recording_song.get_Y_labels()

        y_true_fold_recording, y_pred_fold_recording = recording_Y
        y_true_fold_song, y_pred_fold_song = song_Y
        # print("after the whole evaluation:")
        # print("y_true_fold_song: ", y_true_fold_song)
        # print("y_pred_fold_song: ", y_pred_fold_song)
        # print("y_true_fold_recording: ", y_true_fold_recording)
        # print("y_pred_fold_recording: ", y_pred_fold_recording)
        score_song = round2(accuracy_score(y_true_fold_song, y_pred_fold_song))
        scores_song_list.append(score_song)
        score_song_validation_epoch.append(score_song) # append the final score to the epoch list

        score_recording = round2(accuracy_score(y_true_fold_recording, y_pred_fold_recording))
        scores_recording_list.append(score_recording)
        score_recording_validation_epoch.append(score_recording) # append the final score to the epoch list
        
        # # extend the true and predicted labels for the current fold to the overall lists
        # print("**********************") 
        y_true_all_song.extend(y_true_fold_song)
        y_pred_all_song.extend(y_pred_fold_song)

        y_true_all_recording.extend(y_true_fold_recording)
        y_pred_all_recording.extend(y_pred_fold_recording)
        # print(y_true_all_song)
        # print(y_pred_all_song)
    
        ############################# ############################# ############################# #############################
    
    # take a look of the epoch validation list
    print("score_segment_validation_epoch: \n", score_segment_validation_epoch)
    print("score_recording_validation_epoch: \n", score_recording_validation_epoch)
    print("score_song_validation_epoch: \n", score_song_validation_epoch)

    y_true_all_segment = [int(i) for i in y_true_all_segment]
    y_pred_all_segment = [int(i) for i in y_pred_all_segment]
    
    print("after the whole xvalidation:")
    print("y_true_all_segment: ", y_true_all_segment)
    print("y_pred_all_segment: ", y_pred_all_segment)
    print("y_true_all_recording: ", y_true_all_recording)
    print("y_pred_all_recording: ", y_pred_all_recording)
    print("y_true_all_song: ", y_true_all_song)
    print("y_pred_all_song: ", y_pred_all_song)
    # print their size too
    print("size of y_true_all_segment: ", len(y_true_all_segment))
    print("size of y_pred_all_segment: ", len(y_pred_all_segment))
    print("size of y_true_all_recording: ", len(y_true_all_recording))
    print("size of y_pred_all_recording: ", len(y_pred_all_recording))
    print("size of y_true_all_song: ", len(y_true_all_song))
    print("size of y_pred_all_song: ", len(y_pred_all_song))
          

    print("Xvalidation scores for segment level are:", scores_segment_list)
    
    print("Xvalidation scores for song level are:", scores_song_list)

    print("Xvalidation scores for recording level are:", scores_recording_list)
        
    aggregate_accuracy_segment = round2(accuracy_score(y_true_all_segment, y_pred_all_segment))
    aggregate_accuracy_song = round2(accuracy_score(y_true_all_song, y_pred_all_song))
    aggregate_accuracy_recording = round2(accuracy_score(y_true_all_recording, y_pred_all_recording))
    
    print("aggregate accuracy for segment", aggregate_accuracy_segment)
    print("aggregate accuracy for song", aggregate_accuracy_song)
    print("aggregate accuracy for recording", aggregate_accuracy_recording)

    conf_matrix_segment = confusion_matrix(y_true_all_segment, y_pred_all_segment)
    
    conf_matrix_song = confusion_matrix(y_true_all_song, y_pred_all_song)

    conf_matrix_recording = confusion_matrix(y_true_all_recording, y_pred_all_recording)

    precision_segment, recall_segment, F1_segment = get_metrics(conf_matrix_segment)
    precision_song, recall_song, F1_song = get_metrics(conf_matrix_song)
    precision_recording, recall_recording, F1_recording = get_metrics(conf_matrix_recording)

    print("precision(segment, recording, song): ", precision_segment, precision_recording, precision_song)
    print("recall(segment, recording, song): ", recall_segment, recall_recording, recall_song)
    print("F1(segment, recording, song): ", F1_segment, F1_recording, F1_song)

    if NEPTUNE_SWITCH ==1:
        runtime["info/seg/scores per folds"] = str(scores_segment_list)
        runtime["info/seg/accuracy"] = aggregate_accuracy_segment
        runtime["info/seg/precision"] = precision_segment
        runtime["info/seg/recall"] = recall_segment
        runtime["info/seg/F1"] = F1_segment
        # no need to do segment because it is already being evaluate when getting the score
        # for i in range(len(score_segment_validation_epoch)):
        #     runtime["info/seg/validation_epoch"].append(
        #         value = score_segment_validation_epoch[i],
        #         step = i*Evaluate_Frequency
        #     )
        
        runtime["info/song/scores per folds"] = str(scores_song_list)
        runtime["info/song/accuracy"] = aggregate_accuracy_song
        runtime["info/song/precision"] = precision_song
        runtime["info/song/recall"] = recall_song
        runtime["info/song/F1"] = F1_song
        for i in range(len(score_song_validation_epoch)):
            runtime["info/song/validation_epoch"].append(
                value = score_song_validation_epoch[i],
                step = i*Evaluate_Frequency
            )

        runtime["info/recor/scores per folds"] = str(scores_recording_list)
        runtime["info/recor/accuracy"] = aggregate_accuracy_recording
        runtime["info/recor/precision"] = precision_recording
        runtime["info/recor/recall"] = recall_recording
        runtime["info/recor/F1"] = F1_recording
        for i in range(len(score_recording_validation_epoch)):
            runtime["info/recor/validation_epoch"].append(
                value = score_recording_validation_epoch[i],
                step = i*Evaluate_Frequency
            )

    # save confusion matrix
    save_conf_matrix(conf_matrix_segment, './cfMatrix_seg.png')
    save_normed_conf_matrix(conf_matrix_segment, './cfMatrix_seg_normed.png')
    save_conf_matrix(conf_matrix_song, './cfMatrix_song.png')
    save_normed_conf_matrix(conf_matrix_song, './cfMatrix_song_normed.png')
    save_conf_matrix(conf_matrix_recording, './cfMatrix_recor.png')
    save_normed_conf_matrix(conf_matrix_recording, './cfMatrix_recor_normed.png')

    random_pick_rate = round2(overall_random_pick_times/overall_voting_times) if overall_voting_times != 0 else 0
    random_pick_print_string = str(overall_random_pick_times) + " times out of " + str(overall_voting_times) + ", the percentage is " + str(random_pick_rate)
    if NEPTUNE_SWITCH == 1:
        runtime["info/random vote"] = random_pick_print_string
        runtime["info/random pick rate"] = random_pick_rate
    else:
        print("random vote", random_pick_print_string)

def print_running_information():
    print("running information:")
    print("running on data:", path)
    print("***** ***** *****")
    print("hyperparams:")
    for i in hyperparams:
        print(i, hyperparams[i])
    print("***** ***** *****")

def get_metrics(confusion_matrix):
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]

    precision = round2(TP/(TP+FP))
    recall = round2(TP/(TP+FN))
    F1 = round2(2 * (precision * recall) / (precision + recall))
    return precision, recall, F1

# change the model structure here
def model_adding(model): # will return the optimizer for keeping all the model settings in this one function

    if MODEL == "dummy":
        ##### simple model area ######
        model.add(LSTM(units=1, input_shape=(16, 1024)))
        model.add(Dense(units=1, activation='sigmoid'))
        optim = 'SGD'
        ##### simple model area done ######
    
    if MODEL == "MLP":
        ### MLP
        model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        model.add(Flatten())
        model.add(Dense(8, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optim = 'adam'

    if MODEL == "CNN":
        if method == "pool64":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Conv1D(filters=6, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
        
        if method == "drop0.3":
            ###### try CNN model NO padding ###### 
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Dropout(hyperparams['dropout']))
            model.add(Dense(16, activation='sigmoid'))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Dropout(hyperparams['dropout']))
            
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
            ###### try CNN model NO padding done ######

        if method == "L2e2":
            ###### try CNN model NO padding ###### 
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
            ###### try CNN model NO padding done ######

        if method == "drop0.3+L2e2":
            ###### try CNN model NO padding ###### 
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Dropout(hyperparams['dropout']))
            model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            model.add(Dropout(hyperparams['dropout']))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
            ###### try CNN model NO padding done ######
        if method == "none(oneDense)":
            ###### try CNN model NO padding ###### 
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            # model.add(Dropout(hyperparams['dropout']))
            # model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            # model.add(Dropout(hyperparams['dropout']))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
            ###### try CNN model NO padding done ######
        if method == "L2e2onCNN1CNN2(oneDense)":
            ###### try CNN model NO padding ###### 
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            # model.add(Dropout(hyperparams['dropout']))
            # model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Conv1D(filters=8, kernel_size=64, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(MaxPooling1D(pool_size=64,strides=8))
            # model.add(Dropout(hyperparams['dropout']))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
            ###### try CNN model NO padding done ######

    # if MODEL == "CNNpadding":
    #     if method == "drop0.3":
    #         ###### try CNN model WITH padding ###### Not working current
    #         model.add(Conv1D(filters=8, kernel_size=4, activation='relu', padding='same', input_shape=(16, 1024)))
    #         model.add(MaxPooling1D(pool_size=2))
    #         model.add(Dropout(hyperparams['dropout']))
    #         model.add(Conv1D(filters=4, kernel_size=2, activation='relu', padding='same'))
    #         model.add(MaxPooling1D(pool_size=2))
    #         model.add(Dropout(hyperparams['dropout']))
    #         model.add(Flatten())
    #         model.add(Dense(8, activation='relu'))
    #         model.add(Dense(1, activation='sigmoid'))
    #         optim = 'adam'
    #         ###### try CNN model WITH padding done ######`

    if MODEL == "LSTM":
        if method == "drop0.3":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(LSTM(units=8, return_sequences=True, activation="tanh"))  # Add LSTM layer with 256 units
            model.add(Dropout(0.3))
            model.add(Dense(8, activation="relu"))
            model.add(LSTM(units=6, activation="tanh"))
            model.add(Dropout(0.3))
            model.add(Dense(units=1, activation='sigmoid'))  # Add a dense output layer with sigmoid activation for binary classification
            optim = 'adam'
        
        if method == "L2e2":
            model.add(LSTM(units=8, return_sequences=True, input_shape=(16, 1024), activation="tanh"))  # Add LSTM layer with 256 units
            model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01)))
            model.add(LSTM(units=6, activation="tanh"))
            model.add(Dense(units=1, activation='sigmoid'))  # Add a dense output layer with sigmoid activation for binary classification
            optim = 'adam'
        
        if method == "drop0.3+L2e2":
            model.add(LSTM(units=8, return_sequences=True, input_shape=(16, 1024), activation="tanh"))  # Add LSTM layer with 256 units
            model.add(Dropout(0.3))
            model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01)))
            model.add(LSTM(units=6, activation="tanh"))
            model.add(Dropout(0.3))
            model.add(Dense(units=1, activation='sigmoid'))  # Add a dense output layer with sigmoid activation for binary classification
            optim = 'adam'
    
    if MODEL == "LSTM1":
        if method == "drop0.3":
            _input_shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])
            model.add(LSTM(units=8, return_sequences=True, input_shape=_input_shape, activation="tanh"))  # Add LSTM layer with 8 units
            model.add(Dropout(0.3))
            model.add(LSTM(units=6, activation="tanh"))
            model.add(Dropout(0.3))
            model.add(Dense(units=1, activation='sigmoid'))  # Add a dense output layer with sigmoid activation for binary classification
            optim = 'adam'
        
        if method == "L2e2":
            model.add(LSTM(units=8, return_sequences=True, input_shape=(16, 1024), activation="tanh"))  # Add LSTM layer with 256 units
            model.add(LSTM(units=6, activation="tanh"))
            model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01)))  # Add a dense output layer with sigmoid activation for binary classification
            optim = 'adam'
        
        if method == "drop0.3+L2e2":
            model.add(LSTM(units=8, return_sequences=True, input_shape=(16, 1024), activation="tanh"))  # Add LSTM layer with 256 units
            model.add(Dropout(0.3))
            model.add(LSTM(units=6, activation="tanh"))
            model.add(Dropout(0.3))
            model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01)))  # Add a dense output layer with sigmoid activation for binary classification
            optim = 'adam'


    if MODEL == "Bi_LSTM":
        
        # baseline: two BiLSTM stacked
        # if method == "no_method":
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True)))
        #     model.add(Bidirectional(LSTM(8, return_sequences=False)))
        #     model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
        #     optim = 'adam'

        # if method == 'mydrop0.1':
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True)))
        #     model.add(Dropout(0.1))
        #     model.add(Dense(8, activation="relu"))
        #     model.add(Bidirectional(LSTM(8)))
        #     model.add(Dropout(0.1))
        #     model.add(Dense(1, activation='sigmoid'))
        #     optim = 'adam'
        
        # less one dense layer than parker's
        # if method == "mydrop0.2":
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True)))
        #     model.add(Dropout(0.2))
        #     model.add(Bidirectional(LSTM(8)))
        #     model.add(Dropout(0.2))
        #     model.add(Dense(1, activation='sigmoid'))
        #     optim = 'adam'
        
        # if method == "L2e3onDense":
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True)))
        #     model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
        #     model.add(Bidirectional(LSTM(8, return_sequences=False)))
        #     model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
        #     optim = 'adam'

        # regularizer on middle Dense layer (adapted from parker)
        if method == "L2e2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01)))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
            optim = 'adam'
        
        if method == "drop0.3":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Dense(8, activation="relu"))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dropout(0.3))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
            optim = 'adam'
        
        if method == "drop0.3+L2e2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01)))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dropout(0.3))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
            optim = 'adam'
        
        if method == "16drop0.2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(16, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Dense(8, activation="relu"))
            model.add(Bidirectional(LSTM(16, return_sequences=False)))
            model.add(Dropout(0.2))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
            optim = 'adam'

        # # regularizer on BiLSTM with middle dense layer (new version of l2e2 adapted from parker)
        # if method == "L2e2onBi": 
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
        #     model.add(Dense(8, activation="relu")) # add this to make sure the only different with parker_l2e2 is the regularizer position
        #     model.add(Bidirectional(LSTM(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
        #     model.add(Dense(1, activation='sigmoid'))
        #     optim = 'adam'

        # # old version of l2e2, re-test since the data size was wrong last time
        # if method == "oldL2e2": 
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True, kernel_regularizer=l2(0.01))))
        #     # model.add(Dense(8, activation="relu")) # without this one to make sure this is my old version, no biad regularizer for the same reason
        #     model.add(Bidirectional(LSTM(8, kernel_regularizer=l2(0.01))))
        #     model.add(Dense(1, activation='sigmoid'))
        #     optim = 'adam'

        # if method == "mynorm":
        #     model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(16, 1024)))
        #     model.add(tf.keras.layers.BatchNormalization())
        #     model.add(Bidirectional(LSTM(8)))
        #     model.add(tf.keras.layers.BatchNormalization())
        #     model.add(Dense(1, activation='sigmoid'))
        #     optim = 'adam'

        # if method == "shaun_adapted":
        #     model.add(Bidirectional(LSTM(hyperparams['input_size'], activation='tanh', return_sequences=True)))
        #     model.add(Dropout(0.2))
        #     model.add(Bidirectional(LSTM(8)))
        #     model.add(Dropout(0.2))
        #     model.add(Dense(1, activation='sigmoid'))
        #     optim = 'adam'
        
        # if method == "parker_original":
        #     model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        #     model.add(Bidirectional(LSTM(8, return_sequences=True)))
        #     model.add(Dense(8, activation="relu"))
        #     model.add(Bidirectional(LSTM(8, return_sequences=False), merge_mode="sum"))
        #     model.add(Dense(32, activation="relu", kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3)))
        #     model.add(tf.keras.layers.Flatten())
        #     model.add(Dense(hyperparams["output_size"], activation="softmax", kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3)))
        #     optim = tf.keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.99)

    if MODEL == "Bi_LSTM1":

        if method == "L2e2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            optim = 'adam'
        
        if method == "drop0.3":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dropout(0.3))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
            optim = 'adam'
        
        if method == "drop0.3+L2e2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dropout(0.3))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            optim = 'adam'

    if MODEL == "Bi_LSTM2":

        if method == 'dropout0.1':
            model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(16, 1024)))
            model.add(Dropout(0.1))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.1))
            model.add(Bidirectional(LSTM(8)))
            model.add(Dropout(0.1))
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
        
        if method == "dropout0.2":
            model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(16, 1024)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(8)))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'
        
        if method == "l2":
            model.add(Bidirectional(LSTM(8, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(16, 1024)))
            model.add(Bidirectional(LSTM(8, return_sequences=True, kernel_regularizer=l2(0.01))))
            model.add(Bidirectional(LSTM(8, kernel_regularizer=l2(0.01))))
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'

        if method == "normalize":
            model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(16, 1024)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(Bidirectional(LSTM(8)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(Dense(1, activation='sigmoid'))
            optim = 'adam'

    # return the optimizer for keeping all the model settings in this one function
    return optim



if __name__ == "__main__":
    print_running_information()
    # change model structure in model_adding function
    folds_pattern = get_folds_pattern(fold_count)
    my_x_validation(dataset_of_folds_dictionary, folds_pattern, test_on = TEST_ON) 
                                                                # 0 means using cross validation
                                                                # change this into the only test fold you want to use





    '''code archive'''
    # model.add(Bidirectional(LSTM(hyperparams["input_size"], activation="tanh",return_sequences=True)))
    # model.add(tf.keras.layers.Dropout(hyperparams['dropout']))
    # model.add(Bidirectional(LSTM(int(hyperparams["input_size"]), activation='tanh',return_sequences=True)))
    # model.add(tf.keras.layers.Dropout(hyperparams['dropout']))
    # model.add(Bidirectional(LSTM(int(hyperparams["input_size"]), activation='tanh')))
    # model.add(Dense(hyperparams["output_size"], activation="linear"))

    # optim = tf.keras.optimizers.RMSprop(learning_rate=hyperparams["lr"],momentum=0.5)
    # optim = tf.keras.optimizers.Adam(learning_rate=hyperparams["lr"])
    
    # # no xvalidation test, feed one fold to try if everything is okay
    # model.compile(optimizer='SGD', loss = 'categorical_crossentropy', metrics=["accuracy"])
    # model.build((hyperparams['batch_size'], None, hyperparams['input_size']))
    # dataset = dataset_of_folds_dictionary[1].train # get a fold
    # dataset = dataset.shuffle(buffer_size=1024).batch(hyperparams["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
    # for batch in dataset.train.take(1):
    #     print(batch[0])
    #     print(batch[1])
    # model.fit(dataset.train, epochs=hyperparams['epochs'], verbose = 1)


