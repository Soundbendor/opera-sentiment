import torch
from torch import nn
from torch.utils.data import DataLoader
from HYPERPARAMS import hyperparams
import math
from xvalid_load import folds, folds_size, data_full_dictionary, dataset_of_folds_dictionary, dataset_of_folds_song_level_dictionary

class DummyModel(nn.Module):
    def __init__(self, Method="None"):
        super().__init__()
        _input_size = hyperparams["flatten_size"]
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(_input_size, 1)
        self.relu = nn.ReLU()
        self.output = nn.Linear(1, 1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.output(x)
        predictions = self.softmax(x)
        return predictions
    
class LSTM(nn.Module):
    def __init__(self, Method="None"):
        super().__init__()
        _input_size = hyperparams["input_size"]
        
        self._method = Method
        
        self.lstm1 = nn.LSTM(_input_size, 8, batch_first=True)
        self.tanh1 = nn.Tanh()
        if self._method == "Dropout03":
            self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(8, 6, batch_first=True)
        self.tanh2 = nn.Tanh()
        if self._method == "Dropout03":
            self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layer 1
        out, _ = self.lstm1(x)
        out = self.tanh1(out)
        if self._method == "Dropout03":
            out = self.dropout1(out)

        # LSTM layer 2
        out, _ = self.lstm2(out)
        out = self.tanh2(out)
        if self._method == "Dropout03":
            out = self.dropout2(out)

        # Linear layer
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class CNN(nn.Module):
    def __init__(self, Method="None", cov_kernal = 64, mp_kernal = 32, mp_stride = 8):
        super().__init__()
        _input_size = hyperparams["input_size"]
        self._method = Method
        
        self.conv1 = nn.Conv1d(in_channels=hyperparams["channel_size"], out_channels=8, kernel_size=cov_kernal)
        _last_layer_size = _input_size-cov_kernal+1
        self.relu1 = nn.ReLU()
        if Method == "maxpooling":
            self.pool1 = nn.MaxPool1d(kernel_size = mp_kernal, stride= mp_stride)
            _last_layer_size = math.ceil((_last_layer_size-mp_kernal)/mp_stride)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=6, kernel_size=cov_kernal)
        _last_layer_size = _last_layer_size-cov_kernal+1
        self.relu2 = nn.ReLU()
        if Method == "maxpooling":
            self.pool2 = nn.MaxPool1d(kernel_size=mp_kernal, stride=mp_stride)
            _last_layer_size = math.ceil((_last_layer_size-mp_kernal)/mp_stride)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(6*_last_layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        if self._method == "pool64":
            x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        if self._method == "pool64":
            x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc(x)

        x = self.sigmoid(x)
        return x



if __name__ == "__main__":
    
    from training_time import train
    from Evaluator import Evaluator
    dataset_fold1 = dataset_of_folds_dictionary[1]
    dataset_fold2 = dataset_of_folds_dictionary[2]

    train_loader = DataLoader(dataset_fold1, batch_size=hyperparams["batch_size"], shuffle=True)

    test_loader = DataLoader(dataset_fold2, batch_size=hyperparams["batch_size"], shuffle=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")
    net = CNN("maxpooling").to(device)
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(net.parameters(),
                                lr = hyperparams["lr"])

    EPOCHS = 1

    # train(net, train_loader, loss_fn, optimiser, device, EPOCHS)

    evaluator = Evaluator(net,loss_fn, device)
    accuracy_rec, accuracy_song = evaluator.evaluate_recording_and_song(dataset_of_folds_song_level_dictionary[1])
    print(evaluator.predictions_rec)
    print(evaluator.targets_rec)
    print(evaluator.predictions_song)
    print(evaluator.targets_song)
    print(accuracy_rec)
    print(accuracy_song)