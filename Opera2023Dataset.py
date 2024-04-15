from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import torchaudio
import os
from torch.nn import functional as F

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

if __name__ == '__main__':
    from HYPERPARAMS import hyperparams
    from ENV import REPRESENTATION

    target_class = "emotion_binary"

    csv_file1 = 'trimmed_30_Padding-S/ch/9/wav00/ch_9_wav00.csv'
    file_dir1 = 'trimmed_30_Padding-S/ch/9/wav00'
    # dataset1 = Opera2023Dataset(csv_file1, file_dir1, target_class, hyperparams['input_size'])

    csv_file2 = 'trimmed_30_Padding-S/ch/9/wav01/ch_9_wav01.csv'
    file_dir2 = 'trimmed_30_Padding-S/ch/9/wav01'
    # dataset2 = Opera2023Dataset(csv_file2, file_dir2, target_class, hyperparams['input_size'])
    # # print(dataset[0][0].shape)
    # # data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # # print(data_loader)
    # # for batch in data_loader:
    # #     print(batch[0])
    # #     print(batch[1].shape)
    # print(dataset1[0])


    # # test concat dataset

    # # concat_dataset = ConcatDataset([dataset1, dataset2])
    # # print(concat_dataset)
    # # print(concat_dataset[0][0].shape)

    # ======= Test MelSpec Dataset =========

    # dataset1 = Opera2023Dataset_MelSpec(csv_file1, file_dir1, target_class)
    dataset1 = Opera2023Dataset_Spec(csv_file1, file_dir1, target_class, REPRESENTATION)
    for data in dataset1:
        print(data[0].shape)
    print("load in batch")
    data_loader = DataLoader(dataset1, batch_size=2, shuffle=True)
    for batch in data_loader:
        print(batch[0])
        print(batch[1])
        print(batch[0].shape)
        print(batch[1].shape)

    # dataset2 = Opera2023Dataset_MelSpec(csv_file2, file_dir2, target_class)
    # for data in dataset2:
    #     print(data[0].shape)