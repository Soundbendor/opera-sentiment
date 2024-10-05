import torch
from ENV import REPRESENTATION
def train_one_epoch(model, data_loader, loss_fn, optimizer, device, run=None, npt_logger=None):
    total_correct = 0
    # if REPRESENTATION == "raw+lyrics":
    for batch in data_loader:
        if REPRESENTATION == "raw+lyrics":
            wave, lyrics, targets = batch
            wave, lyrics, targets = wave.to(device), lyrics.to(device), targets.to(device)
            targets = targets.unsqueeze(1)
            optimizer.zero_grad()
            predictions = model(wave, lyrics)
            loss = loss_fn(predictions, targets.float())
            loss.backward()
            optimizer.step()
        else:
            print("load one batch")
            inputs, targets = batch
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
    
    print(f"Loss: {loss.item()}")
    avg_accuracy = total_correct / len(data_loader.dataset)
    if not run is None and not npt_logger is None:
            run[npt_logger.base_namespace]["batch/loss"].append(loss.item())
            run[npt_logger.base_namespace]["batch/acc"].append(avg_accuracy)
    print(f"Accuracy: {avg_accuracy}")

def train(model, data_loader, loss_fn, optimizer, device, epochs, run=None, npt_logger=None, checkpoint_name=None, evaluate_frequency=0):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device, run, npt_logger)
        if checkpoint_name is not None and evaluate_frequency > 0:
            if (i+1) % evaluate_frequency == 0 or (i+1) == epochs:
                torch.save(model.state_dict(), checkpoint_name+f"_{i+1}.pt")

        # save the model for every evaluate_frequency epoch
    print("Training complete")