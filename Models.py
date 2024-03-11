import torch
from torch import nn
from torch.utils.data import DataLoader
from HYPERPARAMS import hyperparams
import math
from xvalid_load import folds, folds_size, data_full_dictionary, dataset_of_folds_dictionary, dataset_of_folds_song_level_dictionary

class DummyModel(nn.Module):
    def __init__(self, Method="None"):
        super().__init__()
        self.flatten = nn.Flatten()
        input_size = (math.ceil(hyperparams["input_length"]/hyperparams["input_size"])*hyperparams["input_size"])
        self.dense = nn.Linear(input_size, 1)
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
        if Method == "Dropout03":
            self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(8, 6, batch_first=True)
        self.tanh2 = nn.Tanh()
        if Method == "Dropout03":
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


from training_time import train
from Evaluator import Evaluator


if __name__ == "__main__":
    dataset_fold1 = dataset_of_folds_dictionary[1]
    dataset_fold2 = dataset_of_folds_dictionary[2]

    train_loader = DataLoader(dataset_fold1, batch_size=hyperparams["batch_size"], shuffle=True)

    test_loader = DataLoader(dataset_fold2, batch_size=hyperparams["batch_size"], shuffle=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")
    net = DummyModel("Dropout03").to(device)
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(net.parameters(),
                                lr = hyperparams["lr"])

    EPOCHS = 1

    train(net, train_loader, loss_fn, optimiser, device, EPOCHS)

    evaluator_segment = Evaluator(net,loss_fn, device)
    segment_accuracy = evaluator_segment.evaluate_segment(test_loader)
    print(evaluator_segment.predictions_seg)
    print(evaluator_segment.targets_seg)