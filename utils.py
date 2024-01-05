import torch
import itertools

# from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def compute_metrics(CM):
    """
    https://www.kaggle.com/code/ajinkyaabhang/implementing-acc-precision-recall-f1-from-scratch/notebook
    """
    # true negative, ... false positive, etc
    tn, tp, fp, fn = CM[0][0], CM[1][1], CM[0][1], CM[1][0]
    acc_score = (tp + tn) / (tp + tn + fp + fn)

    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        print("divided by zero in precision calculation")
        precision = 0  # divide by 0 error

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        print("divided by zero in recall calculation")
        recall = 0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        print("divided by zero in f1 calculation")
        f1 = 0

    return acc_score, precision, recall, f1


THRESHOLD = 0.427  # calculated in colab notebook
def get_preds(probs, threshold=None):
    if threshold==None:
        threshold = THRESHOLD
    probs[probs>threshold] = 1
    probs[probs<=threshold] = 0
    return probs  # preds


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """

    total_loss = 0.0
    CM = 0

    for batch in val_loader:
        val_samples, val_labels = batch['sequence'].to(device), batch['label'].to(device)

        outputs = model(val_samples)
        val_labels = val_labels.reshape(-1, 1).float()

        total_loss += loss_fn(outputs, val_labels).item()  # change tensor to single val

        CM += confusion_matrix(val_labels.cpu().flatten(), get_preds(outputs).cpu().flatten())

        accuracy, precision, recall, f1 = compute_metrics(CM)

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


# Google Bard
# Assign a higher cost to misclassifying minority class instances, making the model prioritize learning from them.

def weighted_binary_cross_entropy(output, target, weights=None):
    "https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114"
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))



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
