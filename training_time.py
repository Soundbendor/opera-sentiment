import torch
from HYPERPARAMS import hyperparams
def train_one_epoch(model, data_loader, loss_fn, optimizer, device, run=None, npt_logger=None):
    total_correct = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(1)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets.float())
        loss.backward()
        optimizer.step()

        rounded_predictions = torch.round(predictions)
        correct = (rounded_predictions == targets).sum().item()
        total_correct += correct
        acc = correct / len(targets)
        if not run is None and not npt_logger is None:
            run[npt_logger.base_namespace]["batch/loss"].append(loss.item())
            run[npt_logger.base_namespace]["batch/acc"].append(acc)
    
    print(f"Loss: {loss.item()}")
    avg_accuracy = total_correct / len(data_loader.dataset)
    print(f"Accuracy: {avg_accuracy}")

def train(model, data_loader, loss_fn, optimizer, device, epochs, run=None, npt_logger=None):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device, run, npt_logger)
    print("Training complete")