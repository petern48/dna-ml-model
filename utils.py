import torch
import itertools

from gensim.models import Word2Vec
# from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.functional import f1_score
from sklearn.metrics import confusion_matrix


def compute_metrics(CM):
    """
    https://www.kaggle.com/code/ajinkyaabhang/implementing-acc-precision-recall-f1-from-scratch/notebook
    """
    # true negative, ... false positive, etc
    tn, tp, fp, fn = CM[0][0], CM[1][1], CM[0][1], CM[1][0]
    acc_score = (tp + tn) / (tp + tn + fp + fn)

    try:
        precision = tp / (tp + fp)
    except:
        precision = 0  # divide by 0 error

    try:
        recall = tp / (tp + fn)
    except:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0

    return acc_score, precision, recall, f1


THRESHOLD = 0.427  # calculated in colab notebook
def get_preds(probs):
    probs[probs>THRESHOLD] = 1
    probs[probs<=THRESHOLD] = 0
    return probs  # preds


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

        CM += confusion_matrix(val_labels.flatten(), get_preds(outputs).flatten())

        acc_score, precision, recall, f1 = compute_metrics(CM)

        total_loss += loss_fn(outputs, val_labels).item()  # change tensor to single val
        n_correct += (get_preds(outputs) == val_labels).sum().item()
        n_total += len(outputs)

    accuracy = n_correct / n_total

    f1_2 = f1_score(total_outputs.flatten(), total_labels.flatten(), task="binary", num_classes=2).item()

    try:
        assert(acc_score == accuracy)
        assert(round(f1, 3) == round(f1_2, 3))
    except:
        print("acc_score", acc_score, "accuracy", accuracy)
        print("f1", f1, "f1_2", f1_2)

    return total_loss, accuracy, precision, recall, f1


# https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/macro_double_soft_f1.py
def macro_double_soft_f1(y, y_hat, reduction='mean'): # Written in PyTorch
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (torch.FloatTensor): targets array of shape (BATCH_SIZE, N_LABELS), including 0. and 1.
        y_hat (torch.FloatTensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar): value of the cost function for the batch
    """

    # dtype = y_hat.dtype
    # y = y.to(dtype)

    # FloatTensor = torch.cuda.FloatTensor
    # y = FloatTensor(y)
    # y_hat = FloatTensor(y_hat)


    tp = (y_hat * y).sum(dim=0) # soft
    fp = (y_hat * (1-y)).sum(dim=0) # soft
    fn = ((1-y_hat) * y).sum(dim=0) # soft
    tn = ((1-y_hat) * (1-y)).sum(dim=0) # soft

    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0

    if reduction == 'none':
        return cost

    if reduction == 'mean':
        macro_cost = cost.mean()
        return macro_cost


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
