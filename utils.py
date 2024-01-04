import torch
import itertools

from gensim.models import Word2Vec
# from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.functional import f1_score
from sklearn.metrics import confusion_matrix


# not in use
def kmer_embedding_encode(sequence, k):
    """
    Look into for word2vec:
    https://github.com/sw1/16s_embeddings/tree/master/code
    Trained: dna2vec: https://github.com/pnpnpn/dna2vec 
    """
    kmers_list = get_kmers(sequence, k)

# not in use
def get_kmers(sequence, k=6):
    """Source: https://www.reddit.com/r/learnpython/comments/16pjoz2/create_kmers_from_a_sequence/"""
    def windowed(iterable, n):
        its = itertools.tee(iterable, n)
        for idx, it in enumerate(its):
            for _ in range(idx):
                next(it)
        return zip(*its)
    
    return [''.join(l) for l in windowed(sequence, k)]
    # working, returns list of substrings


def compute_metrics(CM):
    """
    https://www.kaggle.com/code/ajinkyaabhang/implementing-acc-precision-recall-f1-from-scratch/notebook
    """
    # true negative, ... false positive, etc
    tn, tp, fp, fn = CM[0][0], CM[1][1], CM[0][1], CM[1][0]
    acc_score = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return acc_score, precision, recall, f1


# Not in use
def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.
    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]
    Example output:
        0.75
    """
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """

    total_loss = 0.0
    n_total = 0.0
    n_correct = 0.0
    total_outputs = torch.empty(0).to(device)
    total_labels = torch.empty(0).to(device)
    CM = 0

    for batch in val_loader:
        val_samples, val_labels = batch['sequence'].to(device), batch['label'].to(device)

        outputs = model(val_samples)
        val_labels = val_labels.reshape(-1, 1).float()

        # val_loss = loss_fn(outputs, val_labels).item()  # change tensor to single val
        # val_accuracy = compute_accuracy(outputs, val_labels)
        total_outputs = torch.cat((total_outputs, outputs))
        total_labels = torch.cat((total_labels, val_labels))

        CM += confusion_matrix(val_labels.flatten(), torch.round(outputs).flatten())

        acc_score, precision, recall, f1 = compute_metrics(CM)

        total_loss += loss_fn(outputs, val_labels).item()  # change tensor to single val
        n_correct += (torch.round(outputs) == val_labels).sum().item()
        n_total += len(outputs)

    accuracy = n_correct / n_total

    f1_2 = f1_score(total_outputs.flatten(), total_labels.flatten(), task="binary", num_classes=2).item()
    assert(acc_score == accuracy)
    assert(round(f1, 3) == round(f1_2, 3))

    return total_loss, accuracy, precision, recall, f1
