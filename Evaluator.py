import torch
from HYPERPARAMS import hyperparams
# three-way evaluator
def flatten_and_int_list(nested_list):
    temp_list = []
    for sublist in nested_list:
        temp_list.extend(sublist)
    int_list = [int(x) for x in temp_list]
    return int_list

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
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.targets_seg.extend(targets.cpu().detach().numpy().tolist())
            targets = targets.unsqueeze(1)
            predictions = self.model(inputs)
            rounded_predictions = torch.round(predictions)
            
            self.predictions_seg.extend(rounded_predictions.cpu().detach().numpy().tolist())
            correct = (rounded_predictions == targets).sum().item()
            total_correct += correct
            loss = self.loss_fn(predictions, targets.float())
            acc = correct / len(targets)

            if not self.run is None and not self.npt_logger is None:
                self.run[self.npt_logger.base_namespace]["eval/loss_seg"].append(loss.item())
                self.run[self.npt_logger.base_namespace]["eval/acc_seg"].append(acc)

        avg_accuracy = total_correct / len(data_loader.dataset)
        
        self.predictions_seg = flatten_and_int_list(self.predictions_seg)

        return avg_accuracy
    
    def evaluate_recording(self, data_index):
        pass

    def evaluate_song(self, data_index):
        pass