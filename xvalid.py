import time
import numpy as np
import math
import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

from sklearn.metrics import confusion_matrix, accuracy_score
from xvalid_load import folds, folds_size, data_full_dictionary, dataset_of_folds_dictionary, hyperparams

from dataset import SimpleAudioClassificationDataset

from ENV import Trimmed_PATH as path # dataset path
from ENV import target_second as piece_size

# HOW to get folds? go to "xvalid_load.py" to load or create new ones

from ENV import fold_count

'''Don't Change Here, if change needed, go back to xvalid_load.py to change these
hyperparams['input_size'] = 1024
hyperparams['batch_size'] = 32
hyperparams['output_size'] = 1
'''

# hyperparams['activation'] = "softmax"
hyperparams['dense_units'] = hyperparams['output_size']
hyperparams['dropout'] = 0.2
hyperparams['lr'] = 0.001
hyperparams["loss"] = 'binary_crossentropy' #'categorical_crossentropy'
hyperparams['input_length'] = 16000*piece_size

import sys
MODEL = sys.argv[1]
method = sys.argv[2]
hyperparams['epochs'] = int(sys.argv[3])

# # if manually set
# MODEL = "Bi_LSTM"
# method = 'shaun_adapted'
# hyperparams['epochs'] = 5


NEPTUNE_SWITCH = 1
TEST_ON = 0 # 0 means using cross validation, 1-5 means the only fold to test on
SAVE_MODEL = 1

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
    runtime["sys/tags"].add([str(piece_size)+"s", str(hyperparams['epochs'])+"epochs", MODEL, str(method)])
    runtime["info/folds_to_size"] = folds_size
    # runtime["info/batch_size"] = hyperparams["batch_size"]
    runtime["info/method"] = method
    ###### for Neptune end ######

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
    flat_array = np.concatenate(arrays, axis=0)
    flat_list = flat_array.tolist()
    # flat the flat_list
    flat_list = [item for sublist in flat_list for item in sublist]
    return flat_list

###### archive functions ######
# def ybatch2list(batched_data):
#     batched_data = batched_data.numpy()
#     batched_data.reshape(np.shape(batched_data)[0],)
#     batched_data = batched_data.tolist()
#     y_value_list = [item for sublist in batched_data for item in sublist]
#     return y_value_list

# def bin_prediction(prediction_list):
#     for i in range(len(prediction_list)):
#       if prediction_list[i] >= 0.5:
#         prediction_list[i] = 1.0
#       elif prediction_list[i] < 0.5:
#         prediction_list[i] = 0.0
#     return prediction_list
###### archive functions ######

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
        runtime["cf_matrix"].upload(File(file_name))

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
        runtime["Normed_cf_matrix"].upload(File(file_name))


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
    scores = []
    scores_manually = []
    y_true_all = []
    y_pred_all = []
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
        print("after concatenate:")
        print(dataset_train)
        
        # shuffle and batch train and test data
        dataset_train = shuffle_and_batch(dataset_train)
        dataset_test = shuffle_and_batch(dataset_test)
        print("after batch and shuffle:")
        print(dataset_train)
        print(dataset_test)
        
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
        
        # print(np.shape(dataset_train))
        
        ###### HARD CODE part
        if hyperparams['epochs'] == 1000:
            model_name = MODEL+"_"+method+"_by500_"+str(i)+".h5"


        print("****** training on folds partern", i,"******")
        if NEPTUNE_SWITCH ==1:
          model.fit(dataset_train, epochs = hyperparams['epochs'], callbacks=[neptune_cbk])
        else:
          model.fit(dataset_train, epochs = hyperparams['epochs'])
        
        epochs_str = str(hyperparams['epochs'])
        model_name = MODEL+"_"+method+"_by"+epochs_str+"_"+str(i)+".h5"
            #  model_name method_name  by epochs     fold number 0-4
        # Bi_LSTM_oldl2e2_by100_0.h5 => means using oldl2e2 method, model trained by 100 epochs on the fold 0 (first fold)
        if SAVE_MODEL==1:
            model.save("models/"+model_name)

        print("****** validating on fold", test_index,"******")
        if NEPTUNE_SWITCH ==1:
          score = round2(model.evaluate(dataset_test, callbacks=[neptune_cbk])[1])
        else:
          score = round2(model.evaluate(dataset_test)[1])
        
        scores.append(score)
        
        '''get whole dataset evaluation'''
        # we should have a whole y_real_whole and y_predict_whole
        y_true_fold = []
        y_pred_fold = []
        for batch_x, batch_y in dataset_test:
            # Make predictions on the batch of data
            batch_y_pred = model.predict(batch_x)
            # Threshold the predicted probabilities to obtain predicted labels
            batch_y_pred = (batch_y_pred > 0.5).astype(int)
            # Append the true and predicted labels to the lists for the current fold
            y_true_fold.append(batch_y.numpy())
            y_pred_fold.append(batch_y_pred) #batch_y_pred is already a numpy array
        
        # Convert the lists of true and predicted labels for the current fold to numpy arrays
        y_true_fold = np.concatenate(y_true_fold, axis=0)
        y_pred_fold = np.concatenate(y_pred_fold, axis=0)
        
        scores_manually.append(round2((y_pred_fold == y_true_fold).mean()))
        
        # Append the true and predicted labels for the current fold to the overall lists
        y_true_all.append(y_true_fold)
        y_pred_all.append(y_pred_fold)

        # break #for testing, only one loop
    
    # print(y_true_all)
    # print(type(y_true_all))
    print("Xvalidation scores from evaluate function are:", scores)
    print("the maximum score is:", max(scores))
    print("Xvalidation scores manually calculated are:", scores_manually)
    
    y_pred_all = super_flat(y_pred_all)
    y_true_all = super_flat(y_true_all)
    all_accuracy_manually = accuracy_score(y_true_all, y_pred_all)
    print("all accuracy manually", round2(all_accuracy_manually))

    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    print(conf_matrix)
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    print("TP:",TP, "FP:", FP, "\n", "TN:", TN, "FN:", FN)

    accuracy = round2((TP+TN)/(TP+FP+TN+FN))
    precision = round2(TP/(TP+FP))
    recall = round2(TP/(TP+FN))
    F1 = round2(2 * (precision * recall) / (precision + recall))
    print("*****evaluation from conf_matrix *****")
    print("accuracy is ", accuracy)
    print("precision is ", precision)
    print("recall is", recall)
    print("F1 socre is", F1)

    if NEPTUNE_SWITCH ==1:
        runtime["info/fold_scores"] = scores
        runtime["info/max_score"] = max(scores)
        runtime["info/overall_accuracy"] = accuracy
        runtime["info/precision"] = precision
        runtime["info/recall"] = recall
        runtime["info/F1"] = F1

    # save confusion matrix
    save_conf_matrix(conf_matrix, './cfMatrix.png')
    save_normed_conf_matrix(conf_matrix, './cfMatrix_normed.png')


def print_running_information():
    print("running information:")
    print("running on data:", path)
    print("***** ***** *****")
    print("hyperparams:")
    for i in hyperparams:
        print(i, hyperparams[i])
    print("***** ***** *****")


# change the model structure here
def model_adding(model): # will return the optimizer for keeping all the model settings in this one function

    if MODEL == "dummy":
        ##### simple model area ######
        model.add(LSTM(units=16, input_shape=(16, 1024)))
        model.add(Dense(units=1, activation='sigmoid'))
        optim = 'SGD'
        ##### simple model area done ######
    
    if MODEL == "CNN":
        ###### try CNN model NO padding ###### Not working current
        model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
        model.add(tf.keras.layers.Reshape((math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"],1)))
        model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
        model.add(MaxPooling1D(pool_size=64,strides=8))
        model.add(Dropout(hyperparams['dropout']))

        model.add(Conv1D(filters=8, kernel_size=64, activation='relu'))
        model.add(MaxPooling1D(pool_size=64,strides=8))
        model.add(Dropout(hyperparams['dropout']))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        optim = 'adam'
        ###### try CNN model NO padding done ######

    if MODEL == "LSTM":

        model.add(LSTM(units=8, return_sequences=True, input_shape=(16, 1024), activation="tanh"))  # Add LSTM layer with 256 units
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(hyperparams['dropout']))
        model.add(LSTM(units=6, activation="tanh"))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(Dropout(hyperparams['dropout']))
        model.add(Dense(units=1, activation='sigmoid'))  # Add a dense output layer with sigmoid activation for binary classification
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
            model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dense(hyperparams["output_size"], activation="sigmoid"))
            optim = 'adam'

        if method == "drop0.2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Dense(8, activation="relu"))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dropout(0.2))
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
        
        if method == "drop0.2+L2e2":
            model.add(tf.keras.Input(shape=(math.ceil(hyperparams["input_length"]/hyperparams["input_size"]), hyperparams["input_size"])))
            model.add(Bidirectional(LSTM(8, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Bidirectional(LSTM(8, return_sequences=False)))
            model.add(Dropout(0.2))
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

    if MODEL == "CNNpadding":
        ###### try CNN model WITH padding ######
        model.add(Conv1D(filters=8, kernel_size=4, activation='relu', padding='same', input_shape=(16, 1024)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hyperparams['dropout']))
        model.add(Conv1D(filters=4, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hyperparams['dropout']))
        model.add(Flatten())
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optim = 'adam'
        ###### try CNN model WITH padding done ######`

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


