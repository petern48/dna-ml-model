import torch
from sklearn.preprocessing import LabelBinarizer
import itertools

from gensim.models import Word2Vec
from torcheval.metrics.functional import multiclass_f1_score



bases = ["A", "C", "G", "T"]
lb = LabelBinarizer()
lb.fit_transform(bases)    


def label_encode(sequence):
    """Apply one hot encoding on a string input"""
    return lb.transform(list(sequence))  # numpy array


def kmer_embedding_encode(sequence, k):
    """
    Look into for word2vec:
    https://github.com/sw1/16s_embeddings/tree/master/code
    Trained: dna2vec: https://github.com/pnpnpn/dna2vec 
    """
    kmers_list = get_kmers(sequence, k)



def get_kmers(sequence, k=6):
    """Source: https://www.reddit.com/r/learnpython/comments/16pjoz2/create_kmers_from_a_sequence/"""
    def windowed(iterable, n):
        its = itertools.tee(iterable, n)
        for idx, it in enumerate(its):
            for _ in range(idx):
                next(it)
        return zip(*its)
    
    return [''.join(l) for l in windowed(sequence, k)]

if __name__=="__main__":
    x= get_kmers("abcdef", 2)
    print(x)


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

    total_loss = 0.0
    n_total = 0.0
    n_correct = 0.0
    total_outputs = torch.empty(0).to(device)
    total_labels = torch.empty(0).to(device)

    for batch in val_loader:
        val_samples, val_labels = batch['sequences'].to(device), batch['labels'].to(device)

        outputs = model(val_samples)
        val_labels = val_labels.reshape(-1, 1).float()

        # val_loss = loss_fn(outputs, val_labels).item()  # change tensor to single val
        # val_accuracy = compute_accuracy(outputs, val_labels)
        total_outputs = torch.cat((total_outputs, outputs))
        total_labels = torch.cat((total_labels, val_labels))

        total_loss += loss_fn(outputs, val_labels).item()  # change tensor to single val
        n_correct += (torch.round(outputs) == val_labels).sum().item()
        n_total += len(outputs)

    val_accuracy = n_correct / n_total

    # TODO: f1 score outputs all 0s
    f1_score = multiclass_f1_score(total_outputs.flatten(), total_labels.flatten())

    return total_loss, val_accuracy, f1_score
