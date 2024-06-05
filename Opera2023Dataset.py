from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import torchaudio
import os
from torch.nn import functional as F
from melody_extraction.predict_on_audio import main as get_melody
import numpy as np
import torch
from utilities.yamlhelp import safe_read_yaml
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

class Opera2023DatasetMelody(Dataset):
    def __init__(self, csv_fir, data_dir, target_class, input_size = -1):
        self._csv = pd.read_csv(csv_fir)
        self._dir = data_dir
        self._input_size = input_size
        self._target_class = target_class

    def __len__(self):
        return len(self._csv)
    
    def _get_label(self, idx):
        return self._csv.loc[idx, self._target_class]
    
    def _get_time_melody(self, file_path):
        time_melody = get_melody(filepath=file_path, model_type="vocal", gpu_index=0)
        return time_melody

    def __getitem__(self, idx):
        file_path = os.path.join(self._dir, "in", self._csv.iloc[idx, 0])
        label = self._get_label(idx)
        # check if melody is already cached
        melody_cache_path = os.path.join(self._dir, "melody", self._csv.iloc[idx, 0].replace(".wav", ".txt"))
        if os.path.exists(melody_cache_path):
            melody = np.loadtxt(melody_cache_path)[:, 1]
        else:
            melody = self._get_time_melody(file_path)[:, 1]
        if self._input_size != -1:
            total_samples = len(melody)
            remainder = total_samples % self._input_size
            if remainder != 0:
                pad_length = self._input_size - remainder
                # Pad the waveform tensor with zeros
                melody = np.pad(melody, (0, pad_length))
            # Reshape the waveform tensor
            melody = melody.reshape(-1, self._input_size)
            melody = torch.from_numpy(melody).float()
            # print(melody[0][0])
            # print(type(melody[0][0].item()))
        return melody, label

class Opera2023Dataset(Dataset):
    def __init__(self, csv_fir, data_dir, target_class, input_size = -1):
        self._csv = pd.read_csv(csv_fir)
        self._dir = data_dir
        self._input_size = input_size
        self._target_class = target_class

    def __len__(self):
        return len(self._csv)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self._dir, "in", self._csv.iloc[idx, 0])
        waveform, _ = torchaudio.load(file_path)
        print("load waveform once")
        label = self._get_label(idx)
        if self._input_size != -1: # means we need to reshape the waveform tensor
            total_samples = waveform.shape[1]
            remainder = total_samples % self._input_size
            if remainder != 0:
                pad_length = self._input_size - remainder
                # Pad the waveform tensor with zeros
                waveform = F.pad(waveform, (0, pad_length))

            # Reshape the waveform tensor
            waveform = waveform.view(-1, self._input_size)
        return waveform, label
    
    def _get_label(self, idx):
        return self._csv.loc[idx, self._target_class]

class Opera2023Dataset_Spec(Dataset):
    def __init__(self, csv_fir, data_dir, target_class, spectrogram_type):
        from ENV import SAMPLE_RATE
        self._csv = pd.read_csv(csv_fir)
        self._dir = data_dir
        self._target_class = target_class
        
        if spectrogram_type == "mel":
            self._transformation = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
        elif spectrogram_type == "mfcc":
            self._transformation = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=20,
                melkwargs={"n_mels": 64}
            )
        else:
            raise ValueError("Invalid spectrogram type")
    
    def __len__(self):
        return len(self._csv)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self._dir, "in", self._csv.iloc[idx, 0])
        waveform, _ = torchaudio.load(file_path)
        signal = self._transformation(waveform) # this is assuming all the audio is mono and has the same sample rate, 
                                                # which we already did in the preprocessing
                                                # so no need to resample or convert to mono here
        label = self._get_label(idx)
        return signal, label
    
    def _get_label(self, idx):
        return self._csv.loc[idx, self._target_class]

class Opera2023Dataset_lyrics_bert(Dataset):
    def __init__(self, csv_fir, data_dir, target_class, input_size = -1):
        self._csv = pd.read_csv(csv_fir)
        self._dir = data_dir
        self._input_size = input_size # not used in this representation
        self._target_class = target_class
        def get_lyrics_text(file_path):
            yaml_file_path = file_path.rsplit('/', 1)[0]+"/metadata.yaml"
            metadata = safe_read_yaml(yaml_file_path)
            english_lyrics = metadata['lyric']['english']
            return english_lyrics
        
        self._lyrics = get_lyrics_text(data_dir)

    def __len__(self):
        return len(self._csv)
    
    def _get_label(self, idx):
        return self._csv.loc[idx, self._target_class]
    
    def __getitem__(self, idx):
        self._csv.iloc[idx, 0] # use this to raise a inner exception if the index is out of range
        label = self._get_label(idx)
        text = self._lyrics
        # Tokenize the text
        output = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**output).pooler_output
        return output, label

if __name__ == '__main__':
    from HYPERPARAMS import hyperparams
    from ENV import REPRESENTATION

    target_class = "emotion_binary"

    csv_file1 = 'trimmed_30_Padding-S/ch/10/wav00/ch_10_wav00.csv'
    file_dir1 = 'trimmed_30_Padding-S/ch/10/wav00'
    # dataset1 = Opera2023Dataset(csv_file1, file_dir1, target_class, hyperparams['input_size'])

    csv_file2 = 'trimmed_30_Padding-S/ch/9/wav01/ch_9_wav01.csv'
    file_dir2 = 'trimmed_30_Padding-S/ch/9/wav01'
    # dataset2 = Opera2023Dataset(csv_file2, file_dir2, target_class, hyperparams['input_size'])
    # print("initialization done")
    # for data in dataset2:
    #     print(data[0].shape)
    # # print(dataset[0][0].shape)
    # data_loader = DataLoader(dataset2, batch_size=2, shuffle=True)
    # # print(data_loader)
    # for batch in data_loader:
    #     print(batch[0])
    #     print(batch[1].shape)
    # # print(dataset1[0])


    # # test concat dataset

    # # concat_dataset = ConcatDataset([dataset1, dataset2])
    # # print(concat_dataset)
    # # print(concat_dataset[0][0].shape)

    # ======= Test MelSpec Dataset =========

    # dataset1 = Opera2023Dataset_MelSpec(csv_file1, file_dir1, target_class)
    # dataset_raw = Opera2023Dataset(csv_file1, file_dir1, target_class, hyperparams['input_size'])
    # dataset_spec = Opera2023Dataset_Spec(csv_file1, file_dir1, target_class, REPRESENTATION)
    # if REPRESENTATION == "raw":
    #     dataset1 = dataset_raw
    # else:
    #     dataset1 = dataset_spec
    # for data in dataset1:
    #     print(data[0].shape)
    # print("load in batch")
    # data_loader = DataLoader(dataset1, batch_size=2, shuffle=True)
    # for batch in data_loader:
    #     print(batch[0])
    #     print(batch[1])
    #     print(batch[0].shape)
    #     print(batch[1].shape)

    # dataset2 = Opera2023Dataset_MelSpec(csv_file2, file_dir2, target_class)
    # for data in dataset2:
    #     print(data[0].shape)
        
    # ======= Test Melody Dataset =========
    # dataset_melody = Opera2023DatasetMelody(csv_file2, file_dir2, target_class, 1024)
    # # for data in dataset_melody:
    # #     print(data[0])
    # #     print(data[1])
    # data_loader = DataLoader(dataset_melody, batch_size=2, shuffle=True)
    # for batch in data_loader:
    #     print(batch[0])
    #     print(batch[1])
    #     print(batch[0].shape)
    #     print(batch[1].shape)

    # ======= Test Lyrics Dataset =========
    dataset_lyrics = Opera2023Dataset_lyrics_bert(csv_file1, file_dir1, target_class)
    for data in dataset_lyrics:
        print(data[0])
        print(data[1])
        # pass