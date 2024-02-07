### No need to change this part
### unless you want to change the dataset name or for other custom purpose
# the original dataset name
Data_PATH = "operadataset2023"
Unified_PATH = 'unified'
Trimmed_PATH = 'trimmed'

### hyperparameters related to Machine Learning
### keep them consistent during EVERY workflow steps (from data preprocessing to evaluation)
target_second = 30
evaluation_method =  "song_evaluation" # "segment_evaluation" or "song_evaluation"
segment_method = "Padding" # "Padding" or "Dropping"
fold_count = 5

### !!! MUST NOT Change this part !!!
Trimmed_PATH = Trimmed_PATH + "_" + str(target_second) + "_" +segment_method
### !!! MUST NOT Change this part !!!