from ENV import NEPTUNE_SWITCH
import numpy as np
import tensorflow as tf
import random

# round number to 2 digits
def round2(num):
    return round(num,2)

def voting(prediction_list):
    positive_votes = 0
    negative_votes = 0
    for prediction in prediction_list:
        if prediction == 1:
            positive_votes += 1
        elif prediction == 0:
            negative_votes += 1
        else:
            raise Exception("voting error: prediction is not 0 or 1")
    if positive_votes > negative_votes:
        return 1, False
    elif positive_votes < negative_votes:
        return 0, False
    elif positive_votes == negative_votes:
        # if the votes are equal, return a random value between 0 and 1
        return random.randint(0, 1), True

def flatten_nested_list(nested_list):
    temp_list = []
    for sublist in nested_list:
        temp_list.extend(sublist)
    return temp_list

class Evaluator_Segment:
    def __init__(self, model, data, neptune_cbk=None):
        self.model = model
        self.data = data
        self.neptune_cbk = neptune_cbk

    def get_score(self):
        if NEPTUNE_SWITCH ==1:
            score_segment = round2(self.model.evaluate(self.data, callbacks=[self.neptune_cbk])[1])
        else:
            score_segment = round2(self.model.evaluate(self.data)[1])
        return score_segment
    
    def get_Y_labels(self):
        y_true_fold_segment = []
        y_pred_fold_segment = []
        for batch_x, batch_y in self.data:
            # Make predictions on the batch of data
            batch_y_pred = self.model.predict(batch_x)
            # Threshold the predicted probabilities to obtain predicted labels
            batch_y_pred = (batch_y_pred > 0.5).astype(int)
            # Append the true and predicted labels to the lists for the current fold
            y_true_fold_segment.append(batch_y.numpy())
            y_pred_fold_segment.append(batch_y_pred) #batch_y_pred is already a numpy array
        
        # Convert the lists of true and predicted labels for the current fold to numpy arrays
        y_true_fold_segment = np.concatenate(y_true_fold_segment, axis=0)
        y_pred_fold_segment = np.concatenate(y_pred_fold_segment, axis=0)

        y_true_fold_segment = flatten_nested_list(y_true_fold_segment)
        y_pred_fold_segment = flatten_nested_list(y_pred_fold_segment)
        return y_true_fold_segment, y_pred_fold_segment
    

class Evaluator_Recording_Song:
    def __init__(self, model, data, dataset_of_folds_song_level_dictionary_on_test_fold, neptune_cbk=None):
        self.model = model
        self.data = data
        self.dataset_of_folds_song_level_dictionary_on_test_fold = dataset_of_folds_song_level_dictionary_on_test_fold
        self.neptune_cbk = neptune_cbk
    
    def get_Y_labels(self):
        y_true_fold_song = []
        y_pred_fold_song = []
        y_true_fold_recording = []
        y_pred_fold_recording = []
        for song_id in self.dataset_of_folds_song_level_dictionary_on_test_fold.keys():
            # all data below belongs to one song
            this_song_prediction = []
            for one_recording_data in self.dataset_of_folds_song_level_dictionary_on_test_fold[song_id]:
                this_recording_prediction = []
                for single_segment_data in one_recording_data.train.take(100):
                    single_segment_x = single_segment_data[0]
                    single_segment_y = single_segment_data[1]
                    single_segment_x = tf.expand_dims(single_segment_x, axis=0)
                    single_segment_y_pred = self.model.predict(single_segment_x)
                    single_segment_y_pred = (single_segment_y_pred > 0.5).astype(int)
                    single_segment_y_pred = int(single_segment_y_pred[0, 0]) # convert numpy array to int
                    this_recording_prediction.append(single_segment_y_pred)
                
                # one recording prediction is done, so do two things
                # 1. append the whole recording prediction to the song prediction list
                this_song_prediction.append(this_recording_prediction)
                # 2. vote for recording level prediction and create ture value for recording level
                voting_result, if_random = voting(this_recording_prediction)
                
                y_pred_fold_recording.append(voting_result)
                y_true_fold_recording.append(int(single_segment_y.numpy()[0]))
                '''
                # this_song_prediction is a nested list, each element is a numpy array. If we look into it:
                # eg: 
                # [[0, 0], [0, 0], [1, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 0]]
                # There are three level in this nested list:
                # The very outside represents the whole song (includes multiple recordings)
                # The next level, inner list, represents the recording
                # The smallest single element represents the segment
                # so: after voting for recording level, the result will be:
                # y_pred_fold_recording = [0, 0, 1, 0, 0]
                # after voting for song level, the result will be:
                # 0 (but we will do song voting after evaluate all the recordings in this song)
                '''
            # song evaluation, voting on song level, aka the very outsite level, so we need to flatten the list
            voting_result, if_random = voting(flatten_nested_list(this_song_prediction))
            
            song_pred_y = voting_result
            song_true_y = int(single_segment_y.numpy()[0])
            y_pred_fold_song.append(song_pred_y)
            y_true_fold_song.append(song_true_y)
            # print("check if song level still works")
            # print("y_pred_fold_song:", y_pred_fold_song)
            # print("y_true_fold_song:", y_true_fold_song)
            # print("check the format for recording level")
            # print("y_pred_fold_recording:", y_pred_fold_recording)
            # print("y_true_fold_recording:", y_true_fold_recording)
        
        recording_Y = [y_true_fold_recording, y_pred_fold_recording]
        song_Y = [y_true_fold_song, y_pred_fold_song]
        return recording_Y, song_Y