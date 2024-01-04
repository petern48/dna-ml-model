import torch
import random

import sys
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
# import utils
import pandas as pd


LINES_PER_SEQUENCE = 4
ACCESSIBLE_LABEL = 1
NOT_ACCESSIBLE_LABEL = 0

rand = random.Random(1)
bases = ["A", "C", "G", "T"]
lb = LabelBinarizer()
lb.fit_transform(bases)


def read_data_file(data_file, accessible=True, labeled=True, shuffle=True):

    label = ACCESSIBLE_LABEL if accessible else NOT_ACCESSIBLE_LABEL
    if not labeled:
        label = None

    sequences = []
    labels = []
    ids = []

    with open(data_file, "r") as f:
        while True:
            try:
                id = next(f)[1:].rstrip()   # accessible47239  or seq1
            except StopIteration:
                break
            sequence = ""

            for _ in range(LINES_PER_SEQUENCE):  # read the 4 lines of dna sequence
                sequence += next(f).rstrip()  # Collapse to one sequence

            sequence = lb.transform(list(sequence))  # input a string sequence
            sequence = np.transpose(sequence)

            sequences.append(torch.Tensor(sequence))

            if labeled:
                labels.append(label)
            else:
                ids.append(id)

    if shuffle:
        if labeled:
            sequences, labels = shuffle_lists(sequences, labels)
        else:
            sequences, ids = shuffle_lists(sequences, ids)

    return sequences, labels, ids


def shuffle_lists(list1, list2):
    """Shuffle the two input lists together as one unit"""
    zipped = list(zip(list1, list2))
    rand.shuffle(zipped)
    list1, list2 = zip(*zipped)
    return list(list1), list(list2)


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, sequences, list2, comp=False):
        """Shuffles the two lists together as one unit"""
        # self.accessible_count = self.read_data_file(acc_data_path, accessible=True)
        # self.not_accessible_count = self.read_data_file(not_acc_data_path, accessible=False)
        # self.sequences, self.labels = self.shuffle_lists(self.sequences, self.labels)
        sequences, list2 = shuffle_lists(sequences, list2)
        self.comp = comp
        self.sequences = sequences
        self.list2 = list2  # either ids or labels (depending on if comp or not)


    def __getitem__(self, i):
        """Gets the ith sequence from the dataset"""
        # row = self.df.iloc[i]
        # batch = {
        #     "sequences": row["sequences"],  # now it should be singular
        #     "labels": row["labels"],
        #     "ids": row["ids"]
        # }
        # return self.df.iloc[i].to_dict()
        list2_name = "id" if self.comp else "label"
        return {"sequence": self.sequences[i], list2_name: self.list2[i]}


    def __len__(self):
        return len(self.sequences)
        # return len(self.df.index)


# For testing can just run python dna_dataset.py
# if __name__ == "__main__":
#     DNADataset(TEST_FILE, TEST_FILE)
