import os
import yaml

class Profiler:
    def __init__(self, USING_PATH):
        self.USING_PATH = USING_PATH
        
        if not os.path.exists(self.USING_PATH):
            raise Exception("Path {} does not exist".format(self.USING_PATH))
        # to save running time, generate song level profile along with wav_info list

        self.wav_info = {
            "ch": {},
            "we": {}
        }

        self.wav_list = {
            "ch": [],
            "we": []
        }

        self.song_count = {
            "ch": 0,
            "we": 0
        }

        self.emotion_types = {
            "ch": {},
            "we": {}
        }

        self.singing_types = {
            "ch": {},
            "we": {}
        }

        self.roles = {
            "ch": {},
            "we": {}
        }

        # Song Level Traverse
        for root, dirs, files in os.walk(self.USING_PATH):
            for file in files:
                # we only need to get info from yaml files
                if file.endswith(".yaml"):
                    yaml_path = os.path.join(root, file)
                    
                    with open(yaml_path,"r") as f:
                        meta = yaml.safe_load(f)
                    
                    lan = meta["language"]
                    self.song_count[lan] += 1

                    emotion_type = meta["emotion_binary"]
                    self.emotion_types[lan][emotion_type] = self.emotion_types[lan].get(emotion_type, 0) + 1

                    singing_type = meta["singing_type"]["singing"]
                    self.singing_types[lan][singing_type] = self.singing_types[lan].get(singing_type, 0) + 1

                    role = meta["singing_type"]["role"]
                    self.roles[lan][role] = self.roles[lan].get(role, 0) + 1

                    # get an info dict for each wav file
                    # e.g. "wav00"
                    for wav in meta["files"]:
                        # e.g. "ch/9/wav00.wav"
                        file_name = meta["files"][wav]["file_dir"]
                        if file_name not in self.wav_info[lan]:
                            self.wav_info[lan][file_name] = {}
                            full_path = os.path.join(self.USING_PATH, file_name)
                            if os.path.exists(full_path):
                                self.wav_list[lan].append(file_name)
                            else: # means it is a trimmed work space
                                trim_folder = full_path.split(".")[0]
                                if os.path.exists(trim_folder):
                                    # traverse .wav file in the trimmed folder and add into wav_list[lan]
                                    for root_, dirs_, files_ in os.walk(trim_folder):
                                        for file_ in files_:
                                            if file_.endswith(".wav"):
                                                self.wav_list[lan].append(os.path.join(file_name.split(".")[0], file_))

                        self.wav_info[lan][file_name]["info"] = meta["files"][wav]["info"]
                        self.wav_info[lan][file_name]["singer"] = meta["files"][wav]["singer"]
    
    def full_profile_recording_level(self, if_print_profile=False):
        self.recording_size = {
            "ch": 0,
            "we": 0
        }

        self.a_cappellas = {
            "ch": {
                "true": 0,
                "false": 0
            },
            "we": {
                "true": 0,
                "false": 0
            }
        }

        self.bit_rates = {
            "ch": {},
            "we": {}
        }

        self.channel_number_s = {
            "ch": {},
            "we": {}
        }

        self.sample_rates = {
            "ch": {},
            "we": {}
        }

        self.bio_genders = {
            "ch": {},
            "we": {}
        }

        self.singer_ids = {
            "ch": {},
            "we": {}
        }

        self.singer_levels = {
            "ch": {},
            "we": {}
        }

        self.singer_names = {
            "ch": {},
            "we": {}
        }

        self.len_sum = {
            "ch": 0,
            "we": 0
        }

        self.len_max = {
            "ch": -1,
            "we": -1
        }

        self.len_min = {
            "ch": -1,
            "we": -1
        }

        # Recording Level Traverse
        for lan in self.wav_info:
            for wav in self.wav_info[lan]:
                self.recording_size[lan] += 1
                if self.wav_info[lan][wav]["info"]["if_a_cappella"] == True:
                    self.a_cappellas[lan]["true"] += 1
                elif self.wav_info[lan][wav]["info"]["if_a_cappella"] == False:
                    self.a_cappellas[lan]["false"] += 1
                else:
                    print("ERROR: {} has non-bool if_a_cappella value(s)".format(wav))
                
                bit_rate = self.wav_info[lan][wav]["info"]["bit_rate"]
                self.bit_rates[lan][bit_rate] = self.bit_rates[lan].get(bit_rate, 0) + 1
                
                channel_number = self.wav_info[lan][wav]["info"]["channel_number"]
                self.channel_number_s[lan][channel_number] = self.channel_number_s[lan].get(channel_number, 0) + 1

                sample_rate = self.wav_info[lan][wav]["info"]["sample_rate"]
                self.sample_rates[lan][sample_rate] = self.sample_rates[lan].get(sample_rate, 0) + 1

                bio_gender = self.wav_info[lan][wav]["singer"]["bio_gender"]
                self.bio_genders[lan][bio_gender] = self.bio_genders[lan].get(bio_gender, 0) + 1

                singer_id = self.wav_info[lan][wav]["singer"]["id"]
                self.singer_ids[lan][singer_id] = self.singer_ids[lan].get(singer_id, 0) + 1

                singer_level = self.wav_info[lan][wav]["singer"]["level"]
                self.singer_levels[lan][singer_level] = self.singer_levels[lan].get(singer_level, 0) + 1

                singer_name = self.wav_info[lan][wav]["singer"]["name"]
                self.singer_names[lan][singer_name] = self.singer_names[lan].get(singer_name, 0) + 1      

                len = self.wav_info[lan][wav]["info"]["duration"]
                if self.len_max[lan] == -1 or len > self.len_max[lan]:
                    self.len_max[lan] = len
                if self.len_min[lan] == -1 or len < self.len_min[lan]:
                    self.len_min[lan] = len
                self.len_sum[lan]+=len

        self.len_mean = {
            "ch": self.len_sum["ch"]/self.recording_size["ch"],
            "we": self.len_sum["we"]/self.recording_size["we"]
        }

        def print_profile():
            print("emotion_types: {}".format(self.emotion_types))
            print("singing_types: {}".format(self.singing_types))
            print("roles: {}".format(self.roles))
            print("song_count: {}".format(self.song_count))
            print("if_a_cappellas: {}".format(self.a_cappellas))
            print("number of recording: {}".format(self.recording_size))
            print("max_len: {}".format(self.len_max))
            print("min_len: {}".format(self.len_min))
            print("mean_len: {}".format(self.len_mean))
            print("bit_rates: {}".format(self.bit_rates))
            print("channel_number_s: {}".format(self.channel_number_s))
            print("sample_rates: {}".format(self.sample_rates))
            print("bio_gender: {}".format(self.bio_genders))
            print("singer_id: {}".format(self.singer_ids))
            print("singer_level: {}".format(self.singer_levels))
            print("singer_name: {}".format(self.singer_names))
        
        if if_print_profile == True:
            print_profile()

    def full_profile_segment_level(self, if_print_profile=False):
        self.segment_count = {
            "ch": 0,
            "we": 0
        }
        self.segment_a_cappellas_count = {
            "ch": {
                "true": 0,
                "false": 0
            },
            "we": {
                "true": 0,
                "false": 0
            }
        }
        self.segment_singer_levels = {
            "ch": {},
            "we": {}
        }
        self.segment_singer_gender = {      
            "ch": {},
            "we": {}
        }
        for lan in self.wav_info:
            for wav_path in self.wav_info[lan]:
                # trasfer wav path in to folder path
                segment_folder = wav_path.replace(".wav", "/in")
                # make full path
                segment_folder = os.path.join(self.USING_PATH, segment_folder)
                # count how many wav files under this "segment_folder"
                current_segment = 0
                for root, dirs, files in os.walk(segment_folder):
                    for file in files:
                        if file.endswith(".wav"):
                            self.segment_count[lan]+=1
                            current_segment+=1
                
                if self.wav_info[lan][wav_path]["info"]["if_a_cappella"] == True:
                    self.segment_a_cappellas_count[lan]["true"] += current_segment
                elif self.wav_info[lan][wav_path]["info"]["if_a_cappella"] == False:
                    self.segment_a_cappellas_count[lan]["false"] += current_segment

                self.segment_singer_levels[lan][self.wav_info[lan][wav_path]["singer"]["level"]] \
                    = self.segment_singer_levels[lan].get(self.wav_info[lan][wav_path]["singer"]["level"], 0) \
                        + current_segment

                self.segment_singer_gender[lan][self.wav_info[lan][wav_path]["singer"]["bio_gender"]] \
                    = self.segment_singer_gender[lan].get(self.wav_info[lan][wav_path]["singer"]["bio_gender"], 0) \
                        + current_segment

        def print_profile():
            print("to get song level/recording level profile, use unified or original data set to profile")
            print("segment_count: {}".format(self.segment_count))
            print("a_cappellas_count in segment: {}".format(self.segment_a_cappellas_count))
            print("singer_levels in segment: {}".format(self.segment_singer_levels))
            print("singer_gender in segment: {}".format(self.segment_singer_gender))
        
        if if_print_profile == True:
            print_profile()
    
    def full_profile(self, if_print_profile=False):
        if "trimmed" in self.USING_PATH:
            self.full_profile_segment_level(if_print_profile)
        else:
            self.full_profile_recording_level(if_print_profile)

if __name__ == '__main__':
    from ENV import Data_PATH, Trimmed_PATH
    using_PATH = Data_PATH
    profiler = Profiler(using_PATH)
    profiler.full_profile(if_print_profile=True)
    print(profiler.wav_list)