''' Why this X-validation is tough?
1. We want each song only show up in one fold, to make sure there is no information leakage.
2. We also want each fold has approximately equal, which is X-validation is supposed to be.
3. But each song has different recording count, so it is tough to meet both requirenment 1 and 2,
    not to mention each recording also has different length, after spliting them, their pieces count might have huge different.

Our solution:
    We sacrifice some randomness, to achieve a pseudo-random X-validation.
1. Split all the songs equallt and randomly first, split them into N folds.
2. Check the size of each fold, they must be different.
3. Chek the biggest difference, set a difference_tolerance, then loop:
        4. Move one song (which means all the pieces from this song) from the most big fold into the most small fold.
    until the difference meet the tolerance.
5. Finally we can get N folds with approximately equal count, also make sure each song only split into one fold.
'''

from trim_count import song_id_to_trimed_count
import random
import math
import copy
import numpy as np
import os
from yamlhelp import safe_read_yaml

# Step 1: Split according to song id
def init_folds(fold_count, lan):
    if lan == "all":
        song_count = len(song_id_to_trimed_count["ch"]) + len(song_id_to_trimed_count["we"])
        song_list = list(song_id_to_trimed_count["ch"].keys()) + list(song_id_to_trimed_count["we"].keys())
    else:
        song_count = len(song_id_to_trimed_count[lan])
        song_list = list(song_id_to_trimed_count[lan].keys())
    random.shuffle(song_list)

    # now we can split it into folds, we can make sure one song is only in one fold,
    fold_song_size = math.ceil(song_count/fold_count)

    folds = {}
    # song_id in each fold
    i = 0
    label = 1
    while i < song_count:
        temp = []
        for j in range(fold_song_size):
            if i < song_count:
                temp.append(song_list[i])
                i+=1
        folds[label] = temp
        label+=1
    return folds

# Step 2: get the size(how many actual recording pieces) of a fold
def get_folds_to_size(folds, lan): # => folds_to_size {fold: size}
    folds_to_size = {}
    if lan == "all":
        for label in folds:
            sum = 0
            for i in folds[label]:
                if i in song_id_to_trimed_count["ch"]:
                    sum+=song_id_to_trimed_count["ch"][i]
                else:
                    sum+=song_id_to_trimed_count["we"][i]
            folds_to_size[label] = sum
    else:
        for label in folds:
            sum = 0
            for i in folds[label]:
                sum+=song_id_to_trimed_count[lan][i]
            folds_to_size[label] = sum
    return folds_to_size

# Step 3: find the biggerst diff among all folds
def find_biggest_diff(dict): # => diff, max_key, min_key
    max_ = max(dict.values())
    min_ = min(dict.values())
    for key,value in dict.items():
        if(value == max_):
            max_key = key
        if(value == min_):
            min_key = key
    diff = max_ - min_
    return diff, max_key, min_key

# Step 4: balance loop
def balance_piece(folds, folds_to_size, tolerance, steps, lan, debug=False):
    diff, max_key, min_key = find_biggest_diff(folds_to_size)
    iter = 0

    while diff>tolerance and iter < steps:
        #working on a deep copy first
        folds_copy = copy.deepcopy(folds)
        pick = random.randint(0,len(folds_copy[max_key])-1)
        while (len(folds_copy[max_key])<2):
            max_key = random.randint(1,fold_count)
        move = folds_copy[max_key].pop(pick)
        folds_copy[min_key].append(move)
        #check the size of the copy
        folds_size_temp = get_folds_to_size(folds_copy, lan)
        diff_temp, max_key_temp, min_key_temp = find_biggest_diff(folds_size_temp)
        if diff_temp < diff:
            diff = diff_temp
            max_key = max_key_temp
            min_key_temp = min_key_temp
            folds = folds_copy
            folds_size = folds_size_temp
            
            if debug:
                print("Debug: the current biggest diff:", diff)
        
        iter += 1
    
    return folds

# wrap the function, in order to get folds distribution easier, and make multiple trials
def get_balance_folds(fold_count, lan, trial=5, tolerance=20, steps=5000, save_as = "", debug=False): # ==> new_folds
    new_folds = dict()
    diff = float('inf')
    for i in range(trial):
        folds = init_folds(fold_count, lan)
        
        if debug:
            print("***** ***** Trial "+str(i+1)+" ***** *****")
            print("1. the first random folds distribution:")
            print_folds(folds)
        
        folds_to_size = get_folds_to_size(folds, lan)
        
        if debug:
            print("2. the start size (recording pieces) of each fold:")
            print(folds_to_size)
            diff_temp = find_biggest_diff(folds_to_size)[0]
            print("the biggest diff among start folds is", diff_temp)
            print("**********")
        
        new_folds_temp= balance_piece(folds, folds_to_size, tolerance = tolerance, steps = steps, lan=lan, debug = debug)
        folds_to_size = get_folds_to_size(new_folds_temp, lan)
        
        diff_temp= find_biggest_diff(folds_to_size)[0]
        
        if debug:
            print("3. the result of adjusting fold size:")
            print("current size (recording pieces) of each fold:")
            print(folds_to_size)
            print("current biggest diff is "+str(diff_temp))
            print("current folds distribution:")
            print_folds(new_folds_temp)

        if diff_temp<diff:
            diff = diff_temp
            new_folds = new_folds_temp
            if debug:
                print("folds updated")
        else:
            if debug:
                print("folds NOT updated")
        
        if debug:
            print("***** ***** Trial "+str(i+1)+" ENDS ***** *****")
    
    if save_as != "":
        save_folds(new_folds, save_as)
    
    return new_folds

def print_folds(folds):
    print("****** ****** ******")
    for key, value in folds.items():
        print(key,':',value)
    print("****** ****** ******")

def save_folds(folds, name): # save folds into a numpy variable
    # create variable folder if not exist
    if not os.path.exists("./variable"):
        os.makedirs("./variable")
    # if path exist, add suffix
    name = name + "_"
    suffix = 0
    extension = ".npy"
    path = name + str(suffix) + extension
    while os.path.exists("./variable/"+path):
        # check if the variable are the same
        if np.load("./variable/"+path, allow_pickle=True).item() == folds:
            print("The folds already exist, no need to save again. It is saved as", path)
            return
        suffix+=1
        path = name + str(suffix) + extension
    np.save("./variable/"+path, folds)

def load_folds(file_name):
    folds = np.load("./variable/"+file_name, allow_pickle=True).item()
    return folds

def get_path_folds(path, lan, folds): # getting the path for each fold: {fold#: [path]}
    # input example : folds: {1: [17, 16, 13, 15], 2: [11, 14, 9], 3: [10]}
    fold_to_path = {}
    for fold_id, folds_distri in folds.items():
        fold_to_path[fold_id] = []
        for song_id in folds_distri:
            mother_path = path+'/'+lan+'/'+str(song_id)+'/'
            yaml_path = mother_path + 'metadata.yaml'
            meta_dict = safe_read_yaml(yaml_path)
            for wav in meta_dict["files"]:
                fold_to_path[fold_id].append(mother_path+wav)

    return fold_to_path

def get_song_id_path_folds(path, lan, folds): # getting the song_id and path for each fold: {fold#: [song1: paths, song2: paths]}
    # input example : folds: {1: [17, 16, 13, 15], 2: [11, 14, 9], 3: [10]}
    fold_to_path = {}
    for fold_id, folds_distri in folds.items():
        fold_to_path[fold_id] = {}
        for song_id in folds_distri:
            fold_to_path[fold_id][song_id] = []
            mother_path = path+'/'+lan+'/'+str(song_id)+'/'
            yaml_path = mother_path + 'metadata.yaml'
            meta_dict = safe_read_yaml(yaml_path)
            for wav in meta_dict["files"]:
                fold_to_path[fold_id][song_id].append(mother_path+wav)

    return fold_to_path

if __name__ == "__main__":
    # folds = init_folds(fold_count, lan)
    # print(song_id_to_trimed_count[lan]) if lan != "all" else print(song_id_to_trimed_count)
    # print(folds)
    # folds_to_size = get_folds_to_size(folds, lan)
    # print(folds_to_size)
    # print(find_biggest_diff(folds_to_size))
    # folds, folds_size = balance_piece(folds, folds_to_size, 10, 100, lan, debug=True)
    # print(folds, folds_size)
    from ENV import fold_count, target_second, segment_method
    lan = 'ch'
    balanced_folds = get_balance_folds(fold_count, lan, save_as="ch_folds_"+str(target_second)+"_"+str(segment_method))
    # folds = load_folds("ch_folds_0.npy")
    # print(folds)
    print(balanced_folds)

    # from ENV import Trimmed_PATH as path
    # path2folds = get_path_folds(path, lan, balanced_folds)
    # print(path2folds)