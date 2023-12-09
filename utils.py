import torch

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]
    Example output:
        0.75
    """
    # look into f1 score
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    for batch in val_loader:
        val_samples, val_labels = batch['sequences'].to(device), batch['labels'].to(device)

        outputs = model(val_samples)
        val_labels = val_labels.reshape(-1, 1).float()

        val_loss = loss_fn(outputs, val_labels).item()  # change tensor to single val
        val_accuracy = compute_accuracy(outputs, val_labels)

    return val_loss, val_accuracy