import torch
import random
from constants import *
import sys
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
# import utils
import pandas as pd


# Felt lazy so i copy and pasted rather than reworked the dataset for inheritance
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        random.seed(1)
        self.sequences = []
        self.ids = []

        bases = ["A", "C", "G", "T"]
        self.lb = LabelBinarizer()
        # self.lb = OneHotEncoder
        self.lb.fit_transform(bases)

        count = self.read_data_file(data_path)

        assert(count == len(self.ids) and count == len(self.sequences))

    def read_data_file(self, data_file):
        seq_count = 0
        with open(data_file, "r") as f:
            while True:
                try:
                    id = next(f)[1:].rstrip()   # accessible47239  or seq1
                except StopIteration:
                    break
                seq_count += 1
                sequence = ""

                for _ in range(LINES_PER_SEQUENCE):  # read the 4 lines of dna sequence
                    sequence += next(f).rstrip()  # Collapse to one sequence

                sequence = self.lb.transform(list(sequence))  # input a string sequence
                sequence = np.transpose(sequence)

                self.sequences.append(torch.Tensor(sequence))  # rstrip()
                self.ids.append(id)
        return seq_count


    def __getitem__(self, i):
        """Gets the ith sequence from the dataset"""
        batch = {
            "sequences": self.sequences[i],
            "ids": self.ids[i]
        }
        return batch

    def __len__(self):
        return len(self.sequences)


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, acc_data_path, not_acc_data_path):
        random.seed(1)  # for consistent results
        self.accessible_count = 0
        self.not_accessible_count = 0
        self.sequences = []  # convert to regular lists later? not self.list
        self.labels = []

        bases = ["A", "C", "G", "T"]
        self.lb = LabelBinarizer()
        self.lb.fit_transform(bases)

        self.accessible_count = self.read_data_file(acc_data_path, accessible=True)
        self.not_accessible_count = self.read_data_file(not_acc_data_path, accessible=False)

        assert(self.accessible_count + self.not_accessible_count == len(self.sequences))

        self.sequences, self.labels = self.shuffle_lists(self.sequences, self.labels)

        self.df = pd.DateFrame({
            "sequences":self.sequences,
            "labels":self.labels
        })


    def read_data_file(self, data_file, accessible=True):

        label = ACCESSIBLE_LABEL if accessible else NOT_ACCESSIBLE_LABEL
        seq_count = 0

        with open(data_file, "r") as f:
            while True:
                try:
                    id = next(f)[1:].rstrip()   # accessible47239  or seq1
                except StopIteration:
                    break
                seq_count += 1
                sequence = ""

                for _ in range(LINES_PER_SEQUENCE):  # read the 4 lines of dna sequence
                    sequence += next(f).rstrip()  # Collapse to one sequence

                sequence = self.lb.transform(list(sequence))  # input a string sequence
                sequence = np.transpose(sequence)
                # append the sequence and label

                self.sequences.append(torch.Tensor(sequence))
                self.labels.append(label)
                # self.ids.append(id)

        return seq_count


    def shuffle_lists(self, list1, list2):
        zipped = list(zip(list1, list2))
        random.shuffle(zipped)
        list1, list2 = zip(*zipped)
        return list(list1), list(list2)


    def __getitem__(self, i):
        """Gets the ith sequence from the dataset"""
        row = self.df.iloc[i]
        batch = {
            "sequences": row["sequences"],
            "labels": row["labels"]
        }
        return batch


    def __len__(self):
        # return len(self.labels)
        return len(self.df.index)


# For testing can just run python dna_dataset.py
if __name__ == "__main__":
    DNADataset(TEST_FILE, TEST_FILE)
