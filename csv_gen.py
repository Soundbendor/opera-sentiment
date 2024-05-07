import os
import shutil
from utilities.Profiler import Profiler
from ENV import target_second, target_class, Trimmed_PATH
from utilities.yamlhelp import safe_update_yaml, safe_read_yaml
import sys
import pandas as pd
# generate record for the whole dataset, no matter ch or we
DEBUGMODE = False

if len(sys.argv)>1 and sys.argv[1] == 'debug':
    DEBUGMODE = True

def add_in_folder(path):
    g = os.walk(path)
    for path_cur, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.endswith('wav'):
                if path_cur.endswith('in'):
                    break
                in_path = os.path.join(path_cur, 'in')
                if not os.path.exists(in_path):
                    os.makedirs(in_path)
                break

def move_to_in(path):
    g = os.walk(path)
    move_pair={}
    for path_cur, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.endswith('wav'):
                if path_cur.endswith('in'):
                    break
                wav_path = os.path.join(path_cur, file_name)
                IN_path = os.path.join(path_cur, "in")
                move_pair[wav_path] = IN_path
                
    for wav_path, IN_path in move_pair.items():
        # only do it if the file already is not in "in"
        if not os.path.exists(os.path.join(IN_path, os.path.basename(wav_path))):
            shutil.move(wav_path, IN_path)
        else:
            print("file already exists in in folder: ", wav_path)

def get_wavlist(path): # get .wav files list from any folder
    wavelist = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.join(path, file_name).endswith('.wav'):
                wavelist.append(os.path.join(path, file_name))
            if os.path.join(path, file_name).endswith('.WAV'):
                new_name = file_name.replace('WAV', 'wav')
                os.rename(os.path.join(path, file_name), os.path.join(path, new_name))
                wavelist.append(os.path.join(path, new_name))
    return wavelist

def get_audio_name(str, path): # ./demoarea/singer_1/pos_1 => singer_1_pos_1 // used for getting the data_name
    new_str = str.replace(path+'/','')
    new_str = new_str.replace('/','_')
    return new_str

# generate csv and index files (dataset and dataset.txt), and clear old tf records
def generate_csv_and_index(path):
    piece_size = target_second
    real_path = [] # song location path: eg: trimmed_demo_15/ch/10/wav06
    g = os.walk(path)
    for path_cur, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.endswith('wav'):
                father_path = path_cur.replace('/in','')
                real_path.append(father_path)
                break

    for path_curr in real_path:
        
        dataset_name = get_audio_name(path_curr, path)
        wavelist = get_wavlist(path_curr)
        # print(dataset_name)
        if len(wavelist) == 0:
            print(dataset_name, 'is EMPTY')
            return
        
        csv_file = os.path.join(path_curr,(dataset_name+'.csv'))
        index_file = os.path.join(path_curr,(dataset_name+'.txt'))
        
        if os.path.exists(csv_file):
            os.remove(csv_file)
    
        if os.path.exists(index_file):
            os.remove(index_file)
        
        if DEBUGMODE:
            print("making csv file: ", csv_file)
        
        data = {
            "name":[],
            "emotion_binary":[],
            "bio_gender":[],
            "level":[],
            "role":[],
            "acappella":[],
            "jingju":[]
        }
        for name in wavelist:
            # eg:
            # name: "trimmed_demo_15/we/2/wav01/in/wav01_003.wav"
            # pure_name: "wav01_003.wav"
            # we need to get the class from yaml file, then 'text.write(pure_name+',0\n')'

            # get yaml path: eg: trimmed_demo_15/we/2/metadata.yaml

            song_dir = os.path.dirname(path_curr)
            yaml_path = os.path.join(song_dir, 'metadata.yaml')
            pure_name = name.split('/')[-1] # eg: wav06_002.wav
            
            data["name"].append(pure_name)
            # get class label from yaml file
            yaml_dict = safe_read_yaml(yaml_path)
            label = yaml_dict['emotion_binary']
            
            if label != 0 and label != 1:
                raise Exception("wrong label found in yaml file: ", yaml_path)
            
            data["emotion_binary"].append(label)
            
            wav_id = pure_name.split("_")[0] # eg: wav06
            yaml_dict = safe_read_yaml(yaml_path)
            label = yaml_dict['files'][wav_id]['singer']['bio_gender']
            if label == 'mal':
                label = 1
            elif label == 'fem':
                label = 0
            elif label != 'mal' and label != 'fem':
                raise Exception("wrong label found in yaml file: ", yaml_path)
            
            data['bio_gender'].append(label)
            
            wav_id = pure_name.split("_")[0] # eg: wav06
            yaml_dict = safe_read_yaml(yaml_path)
            label = yaml_dict['files'][wav_id]['singer']['level']
            if label == 'professional':
                label = 1
            elif label == 'mixed':
                label = 1
            elif label == 'amateur':
                label = 0
            elif label != 'pro' and label != 'amateur' and label != 'mixed':
                raise Exception("wrong label found in yaml file: ", yaml_path)

            data['level'].append(label)
            
            yaml_dict = safe_read_yaml(yaml_path)
            label = yaml_dict["singing_type"]['role']
            if label == "sheng":
                label = 1
            elif label == "dan":
                label = 0
            elif label != "sheng" and label != "dan":
                # don't raise exception if its a western song
                # only raise exceptino if its a chinese song
                if name.split('/')[1] == 'ch':
                    raise Exception("wrong label found in yaml file: ", yaml_path)
                else:
                    label = -1 # work as a flag to indicate that this is a western song
                    print("WARNING: generating role class for western songs, they are not ready for training")
            
            data['role'].append(label)

            wav_id = pure_name.split("_")[0] # eg: wav06
            yaml_dict = safe_read_yaml(yaml_path)
            label = yaml_dict['files'][wav_id]['info']['if_a_cappella']
            if label == True:
                label = 1
            elif label == False:
                label = 0
            elif type(label) != bool:
                raise Exception("wrong label found in yaml file: ", yaml_path)
            
            data['acappella'].append(label)

            yaml_dict = safe_read_yaml(yaml_path)
            label = yaml_dict["singing_type"]['singing']
            if label == "jingju":
                label = 1
            else:
                label = 0

            data['jingju'].append(label)

        if not DEBUGMODE:
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
        else:
            print("the csv data: ", data)

        size = str(len(wavelist))
        length = str(piece_size)

        if not DEBUGMODE:
            # making the real index txt file
            text = open(index_file, 'w')
        else:
            print("making index file: ", index_file)
        index_text = '{"size": '+size+', "length": '+length+', "sample_rate": 16000, "input_lower": 0, "input_upper": 0, "waves": "a", "mods": "a"}'
        
        if not DEBUGMODE:
            text.write(index_text)
            text.close()
        else:
            print("writing: ", index_text)

# this function looks like useless, delete it later if it is true
# def get_datainfo(dataset_path): # get matching data_path and data_name in the dataset path
#     data_info_dict = {} # data_path: data_name
#     g = os.walk(dataset_path)
#     for path_cur, dir_list, file_list in g:
#         for folder_name in dir_list:
#             if folder_name == 'in':
#                 data_path = path_cur
#                 data_name = get_audio_name(path_cur, dataset_path)
#                 data_info_dict[data_path] = data_name
#     return data_info_dict

if __name__ == "__main__":
    if DEBUGMODE:
        print("DEBUG MODE, no record is being generated, no csv or index will be created. Result will be printed out.")
    
    # add in forlder for a brand new trimmed data folder
    add_in_folder(Trimmed_PATH)
    
    # move all wav files into "in" folder
    move_to_in(Trimmed_PATH)
    
    # generate/regenerate csv and index files (dataset and dataset.txt), and clear old tf records
    generate_csv_and_index(Trimmed_PATH)