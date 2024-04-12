import torch
from torch import nn
from torch.utils.data import DataLoader
from HYPERPARAMS import hyperparams
import math
from xvalid_load import folds, folds_size, data_full_dictionary, dataset_of_folds_dictionary, dataset_of_folds_song_level_dictionary

class DummyModel(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        _input_size = hyperparams["flatten_size"]
        
        self.flatten = nn.Flatten()
        self.output = nn.Linear(_input_size, 1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.output(x)
        predictions = self.softmax(x)
        return predictions

class MLP(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hyperparams['flatten_size'], 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class LSTM(nn.Module):
    def __init__(self, input_size, Method="None", Bidirectional=False):
        super().__init__()
        _input_size = hyperparams["input_size"]
        
        self._method = Method
        
        self.lstm1 = nn.LSTM(_input_size, 8, batch_first=True, bidirectional=Bidirectional)
        self.tanh1 = nn.Tanh()
        if self._method == "Dropout03":
            self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(8*2 if Bidirectional else 8, 6, batch_first=True, bidirectional=Bidirectional)
        self.tanh2 = nn.Tanh()
        if self._method == "Dropout03":
            self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(6*2 if Bidirectional else 6, 1)
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

class CNN1D_raw(nn.Module):
    def __init__(self, input_size, Method="None", cov_kernal = 64, mp_kernal = 32, mp_stride = 8):
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
        if self._method == "maxpooling":
            x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        if self._method == "maxpooling":
            x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc(x)

        x = self.sigmoid(x)
        return x

class CNN2D(nn.Module):
    def __init__(self, input_size, Method="None", kernel_size = 5, stride = 1, padding = 2, pool_size = 2):
        super().__init__()
        # the first element of input_size is the batch size so we ignore it
        in_channels = input_size[1]
        h = input_size[2]
        w = input_size[3]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = 32, 
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size)
        )

        h = int((((h-kernel_size+2*padding)/stride)+1)//pool_size)
        w = int((((w-kernel_size+2*padding)/stride)+1)//pool_size)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size)
        )

        h = int((((h-kernel_size+2*padding)/stride)+1)//pool_size)
        w = int((((w-kernel_size+2*padding)/stride)+1)//pool_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size)
        )

        h = int((((h-kernel_size+2*padding)/stride)+1)//pool_size)
        w = int((((w-kernel_size+2*padding)/stride)+1)//pool_size)

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 128, 
                out_channels = 256, 
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size)
        )

        h = int((((h-kernel_size+2*padding)/stride)+1)//pool_size)
        w = int((((w-kernel_size+2*padding)/stride)+1)//pool_size)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*w*h, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class BiLSTM(LSTM):
    def __init__(self, Method="None"):
        super().__init__(Method, Bidirectional=True)


if __name__ == "__main__":
    
    from training_time import train
    from Evaluator import Evaluator
    from ENV import REPRESENTATION
    dataset_fold1 = dataset_of_folds_dictionary[1]
    dataset_fold2 = dataset_of_folds_dictionary[2]

    train_loader = DataLoader(dataset_fold1, batch_size=hyperparams["batch_size"], shuffle=True)

    test_loader = DataLoader(dataset_fold2, batch_size=hyperparams["batch_size"], shuffle=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")
    MEL_input_size = (-1, 1, 64, 938)
    MFCC_input_size = (-1, 1, 20, 2401)

    net_mel = CNN2D(input_size=MEL_input_size).to(device)
    net_mfcc = CNN2D(input_size=MFCC_input_size).to(device)
    
    
    if REPRESENTATION == "mel":
        net = net_mel
    elif REPRESENTATION == "mfcc":
        net = net_mfcc
    
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(net.parameters(),
                                lr = hyperparams["lr"])

    EPOCHS = 1

    train(net, train_loader, loss_fn, optimiser, device, EPOCHS)

    # evaluator = Evaluator(net,loss_fn, device)
    # accuracy_rec, accuracy_song = evaluator.evaluate_recording_and_song(dataset_of_folds_song_level_dictionary[1])
    # print(evaluator.predictions_rec)
    # print(evaluator.targets_rec)
    # print(evaluator.predictions_song)
    # print(evaluator.targets_song)
    # print(accuracy_rec)
    # print(accuracy_song)