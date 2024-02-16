### No need to change this part
### unless you want to change the dataset name or for other custom purpose
# the original dataset name
Data_PATH = "operadataset2023"
Unified_PATH = 'unified'
Trimmed_PATH = 'trimmed'

### hyperparameters related to Machine Learning
### keep them consistent during EVERY workflow steps (from data preprocessing to evaluation)
target_second = 30
evaluation_method =  "segment_evaluation" # "segment_evaluation" or "song_evaluation"
segment_method = "Padding-S" # "Padding-S(ilence)" or "Padding-C(ircular)" or "Dropping"
target_class = "bio_gender"
fold_count = 5

target_class_dictionary = {
    "emotion_binary": "Emo",
    "bio_gender": "Gen",
    "level": "Lev",
}

target_class_short = target_class_dictionary[target_class]

### !!! MUST NOT Change this part !!!
Trimmed_PATH = Trimmed_PATH + "_" + str(target_second) + "_" +segment_method+ "_" + target_class_short
### !!! MUST NOT Change this part !!!