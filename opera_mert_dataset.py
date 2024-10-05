from torch.utils.data import Dataset
import pandas as pd
import os
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torchaudio
import torchaudio.transforms as T

model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)

class opera_mert_dataset(Dataset):
    def __init__(self, csv_fir, data_dir, target_class, input_size = -1):
        self._csv = pd.read_csv(csv_fir)
        self._dir = data_dir
        self._input_size = input_size
        self._target_class = target_class

    def __len__(self):
        return len(self._csv)
    
    def _get_label(self, idx):
        return self._csv.loc[idx, self._target_class]

    def _mert(self, audio_path):
        audio_input, sampling_rate = torchaudio.load(audio_path)
        audio_input = audio_input.squeeze().numpy()
        resample_rate = processor.sampling_rate
        if resample_rate != sampling_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
            audio_input = resampler(torch.from_numpy(audio_input))
            audio_input = audio_input.squeeze().numpy()
        else:
            audio_input = audio_input
        
        inputs = processor(audio_input, sampling_rate=resample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        return time_reduced_hidden_states

    def __getitem__(self, idx):
        audio_path = os.path.join(self._dir, "in", self._csv.iloc[idx, 0])
        label = self._get_label(idx)
        audio_embedding = self._mert(audio_path)
        return audio_embedding, label


if __name__ == '__main__':
    csv_fir = (
        "/nfs/guille/eecs_research/soundbendor/shengxuan/opera2324/opera-singing-dataset/trimmed_30_Padding-S/ch/9/wav00/ch_9_wav00.csv"
    )
    data_dir = (
        "/nfs/guille/eecs_research/soundbendor/shengxuan/opera2324/opera-singing-dataset/trimmed_30_Padding-S/ch/9/wav00"
    )
    target_class = "emotion_binary"

    dataset = opera_mert_dataset(csv_fir, data_dir, target_class)
    print(dataset[0])
    print(len(dataset))
    print(dataset[0][0].shape)

