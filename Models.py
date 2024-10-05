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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.output(x)
        predictions = self.sigmoid(x)
        return predictions

class FC_for_bert(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        self.flatten_size = 1
        for i in input_size[1:]:
            self.flatten_size *= i
        self.flatten = nn.Flatten()
        self.output = nn.Linear(self.flatten_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.output(x)
        predictions = self.sigmoid(x)
        return predictions

class LSTM_fuse_bert(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        self._lstm_input_size = input_size[0][2]
        self._method = Method

        self.lstm_output = 8
        
        self.lstm = nn.LSTM(self._lstm_input_size, self.lstm_output, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(469*self.lstm_output+768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave, lyrics_feature):
        wave_feature, _ = self.lstm(wave)
        wave_feature = self.dropout(wave_feature) # (batch_size, 469, 64)
        wave_feature_flat = wave_feature.reshape(-1, 469*self.lstm_output) # (batch_size, 469*64)
        lyrics_feature = lyrics_feature.squeeze(1) # (batch_size, 768)
        feature = torch.cat((wave_feature_flat, lyrics_feature), dim=1) # (batch_size, 469*64+768)
        predictions = self.fc(feature)
        predictions = self.sigmoid(predictions)
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
    def __init__(self, input_size, Method="None", network_size=4, cov_kernal = 3, mp_kernal = 2):
        super().__init__()
        if network_size < 1 or network_size > 4:
            raise ValueError("Network size must be between 1 and 4")
        _input_size = hyperparams["input_size"]
        self._last_layer_size = _input_size
        self._network_size = network_size
        def conv1d_with_pooling(in_channels, out_channels):
            self._last_layer_size = (self._last_layer_size-cov_kernal+1)//mp_kernal
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, cov_kernal),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=mp_kernal)
            )
        
        out_channels = 32
        self.cwp1 = conv1d_with_pooling(hyperparams["channel_size"], out_channels) # cwp stands for conv1d with pooling
        network_size -= 1
        if network_size > 0:
            out_channels = out_channels*2
            self.cwp2 = conv1d_with_pooling(32, out_channels)
            network_size -= 1
        if network_size > 0:
            out_channels = out_channels*2
            self.cwp3 = conv1d_with_pooling(64, out_channels)
            network_size -= 1
        if network_size > 0:
            out_channels = out_channels*2
            self.cwp4 = conv1d_with_pooling(128, out_channels)
            network_size -= 1
        print("out_channels is", out_channels)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels*self._last_layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cwp1(x)
        if self._network_size > 1:
            x = self.cwp2(x)
        if self._network_size > 2:
            x = self.cwp3(x)
        if self._network_size > 3:
            x = self.cwp4(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
        
class CNN1D_mert(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        self.channel_size = input_size[1]
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.channel_size, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout3 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(12032, 1) # hard code for now
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class MobileNet1DV1(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        input_channel = input_size[1]

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv1d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True),
                )
    
        self.model = nn.Sequential(
                conv_bn(input_channel, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
                nn.AdaptiveAvgPool1d(1)
            )
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        print(x.shape)
        x = self.model(x)
        print(x.shape)
        x = x.view(-1, 1024)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x = self.sigmoid(x)
        print(x.shape)
        print("*****done*****")
        return x

class MobileNet1Dsimple(nn.Module):
    def __init__(self, input_size, Method="None"):
        super().__init__()
        input_channel = input_size[1]

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv1d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True),
                )

        if Method == "none":
            self.model = nn.Sequential(
                    conv_bn(input_channel, 32, 2),
                    conv_dw(32, 64, 1),
                    conv_dw(64, 128, 2),

                    conv_dw(128, 128, 1),
                    conv_dw(128, 256, 2),

                    conv_dw(256, 256, 1),
                    conv_dw(256, 512, 2),
                    
                    conv_dw(512, 512, 1),
                    conv_dw(512, 1024, 2),

                    conv_dw(1024, 1024, 1),
                    
                    nn.AdaptiveAvgPool1d(1)
                )
            self.fc = nn.Linear(1024, 1)
        
        elif Method == "smaller":
            self.model = nn.Sequential(
                    conv_bn(input_channel, 32, 2),
                    conv_dw(32, 64, 1),
                    conv_dw(64, 128, 2),

                    conv_dw(128, 128, 1),
                    conv_dw(128, 256, 2),

                    conv_dw(256, 256, 1),
                    
                    nn.AdaptiveAvgPool1d(1)
                )
            self.fc = nn.Linear(256, 1)
        else:
            raise ValueError("Method not found")
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        print(x.shape)
        x = self.model(x)
        print(x.shape)
        x = x.view(-1, 256)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x = self.sigmoid(x)
        print(x.shape)
        print("*****done*****")
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
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x = self.sigmoid(x)
        print(x.shape)
        return x

class BiLSTM(LSTM):
    def __init__(self, Method="None"):
        super().__init__(Method, Bidirectional=True)


if __name__ == "__main__":
    
    from TorchPipeline.training_time import train
    from TorchPipeline.Evaluator import Evaluator
    from ENV import REPRESENTATION
    from torchinfo import summary
    dataset_fold1 = dataset_of_folds_dictionary[1]
    dataset_fold2 = dataset_of_folds_dictionary[2]

    train_loader = DataLoader(dataset_fold1, batch_size=hyperparams["batch_size"], shuffle=True)

    test_loader = DataLoader(dataset_fold2, batch_size=hyperparams["batch_size"], shuffle=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")
    raw_input_size = (hyperparams["batch_size"], 469, 1024)
    MEL_input_size = (hyperparams["batch_size"], 1, 64, 938)
    MFCC_input_size = (hyperparams["batch_size"], 1, 20, 2401)
    lyrics_bert_input_size = (hyperparams["batch_size"], 1, 768)
    raw_and_lyrics_input_size = [raw_input_size, lyrics_bert_input_size]

    net_raw = CNN1D_raw(input_size=hyperparams["input_size"], network_size=2).to(device)
    # net_raw = MobileNet1DV1(input_size=raw_input_size).to(device)
    net_mel = CNN2D(input_size=MEL_input_size).to(device)
    net_mfcc = CNN2D(input_size=MFCC_input_size).to(device)
    net_fuse = LSTM_fuse_bert(input_size=raw_and_lyrics_input_size).to(device)

    if REPRESENTATION == "raw":
        net = net_raw
        input_size = raw_input_size
    elif REPRESENTATION == "mel":
        net = net_mel
        input_size = MEL_input_size
    elif REPRESENTATION == "mfcc":
        net = net_mfcc
        intput_size = MFCC_input_size
    elif REPRESENTATION == "raw+lyrics":
        net = net_fuse
        input_size = raw_and_lyrics_input_size
    
    print(input_size)

    model_summary = summary(net, input_size, device=device)

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