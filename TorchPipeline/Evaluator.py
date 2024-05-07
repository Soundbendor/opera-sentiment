import torch
from torch.utils.data import DataLoader
import random
# three-way evaluator
def flatten_and_int_list(nested_list):
    temp_list = []
    for sublist in nested_list:
        temp_list.extend(sublist)
    int_list = [int(x) for x in temp_list]
    return int_list

def voting(prediction_list):
    positive_votes = 0
    negative_votes = 0
    for prediction in prediction_list:
        if prediction == 1:
            positive_votes += 1
        elif prediction == 0:
            negative_votes += 1
        else:
            raise Exception("voting error: prediction is not 0 or 1")
    if positive_votes > negative_votes:
        return 1, False
    elif positive_votes < negative_votes:
        return 0, False
    elif positive_votes == negative_votes:
        # if the votes are equal, return a random value between 0 and 1
        return random.randint(0, 1), True

class Evaluator:
    def __init__(self, model, loss_fn, device, run=None, npt_logger=None):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.run = run
        self.npt_logger = npt_logger

    def evaluate_segment(self, data_loader):
        self.predictions_seg = []
        self.targets_seg = []

        self.model.eval()
        total_correct = 0
        for inputs, targets in data_loader:
            # each loop is a batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.targets_seg.extend(targets.cpu().detach().numpy().tolist())
            targets = targets.unsqueeze(1)
            with torch.no_grad():
                predictions = self.model(inputs)
            rounded_predictions = torch.round(predictions)
            
            self.predictions_seg.extend(rounded_predictions.cpu().detach().numpy().tolist())
            correct = (rounded_predictions == targets).sum().item()
            total_correct += correct
            loss = self.loss_fn(predictions, targets.float())
            acc = correct / len(targets)

        # accuracy of the entire dataset
        accuracy_all = total_correct / len(data_loader.dataset)
        if not self.run is None and not self.npt_logger is None:
            self.run["result/seg/loss_eval"].append(loss.item())
            self.run["result/seg/acc_eval"].append(acc)
        
        self.predictions_seg = flatten_and_int_list(self.predictions_seg)

        return accuracy_all
    
    def evaluate_recording_and_song(self, data_dict):
        self.predictions_rec = []
        self.targets_rec = []
        self.predictions_song = []
        self.targets_song = []

        self.model.eval()
        for song_id in data_dict.keys():
            # all data below belongs to one song
            this_song_predictions = []
            for one_recording_data in data_dict[song_id]:
                this_recording_predictions = []
                # one_recording_data is the data before putting into data loader
                one_recording_loader = DataLoader(one_recording_data, batch_size=1, shuffle=False)
                                                                    # batch size is 1 because we want to evaluate one piece at a time
                                                                    # shuffle is False because we want to keep the order of the pieces
                for inputs, targets in one_recording_loader:
                    # each loop is a segment
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    with torch.no_grad():
                        predictions = self.model(inputs)
                    rounded_predictions = torch.round(predictions)
                    this_recording_predictions.extend(rounded_predictions.cpu().detach().numpy().tolist())
                    this_song_predictions.extend(rounded_predictions.cpu().detach().numpy().tolist())
                    
                this_recording_predictions = flatten_and_int_list(this_recording_predictions)
                voting_recording, is_random = voting(this_recording_predictions)
                self.predictions_rec.append(voting_recording)
                
                # append the last target to list
                self.targets_rec.append(targets.cpu().detach().numpy().tolist())
            this_song_predictions = flatten_and_int_list(this_song_predictions)
            voting_song, is_random = voting(this_song_predictions)
            self.predictions_song.append(voting_song)
            self.targets_song.append(targets.cpu().detach().numpy().tolist())
            
        self.targets_rec = flatten_and_int_list(self.targets_rec)
        self.targets_song = flatten_and_int_list(self.targets_song)


        accuracy_rec = sum([1 for i in range(len(self.predictions_rec)) if self.predictions_rec[i] == self.targets_rec[i]]) / len(self.predictions_rec)
        accuracy_song = sum([1 for i in range(len(self.predictions_song)) if self.predictions_song[i] == self.targets_song[i]]) / len(self.predictions_song)
        
        if not self.run is None and not self.npt_logger is None:
            self.run["result/rec/acc_eval"].append(accuracy_rec)
            self.run["result/song/acc_eval"].append(accuracy_song)
        
        return accuracy_rec, accuracy_song
