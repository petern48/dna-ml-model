import torch
import random
from constants import *
import sys
from sklearn.preprocessing import LabelBinarizer
import numpy as np

class DNADataset(torch.utils.data.Dataset):
    def __init__(self, acc_data_path, not_acc_data_path):
        random.seed(1)  # for consistent results
        self.accessible_count = 0
        self.not_accessible_count = 0
        self.sequences = []
        self.labels = []
        self.ids = []

        bases = ["A", "C", "G", "T"]
        self.lb = LabelBinarizer()
        self.lb.fit_transform(bases)

        self.accessible_count = self.read_data_file(acc_data_path, accessible=True)
        self.not_accessible_count = self.read_data_file(not_acc_data_path, accessible=False)

        assert(self.accessible_count + self.not_accessible_count == len(self.sequences))

        self.sequences, self.labels = self.shuffle_lists(self.sequences, self.labels)


    def read_data_file(self, data_file, accessible=True):

        label = ACCESSIBLE_LABEL if accessible else NOT_ACCESSIBLE_LABEL
        seq_count = 0

        with open(data_file, "r") as f:
            while True:
                try:
                    id = next(f).rstrip()   # >accessible47239
                except StopIteration:
                    break
                seq_count += 1
                sequence = ""


                for _ in range(LINES_PER_SEQUENCE):  # read the 4 lines of dna sequence
                    sequence += next(f).rstrip()  # Collapse to one sequence

                sequence = self.label_encode(sequence)  # input a string sequence
                sequence = np.transpose(sequence)
                # append the sequence and label

                self.sequences.append(torch.Tensor(sequence))  #rstrip()
                self.labels.append(label)
                self.ids.append(id)
        
        return seq_count

    
    def shuffle_lists(self, list1, list2):
        zipped = list(zip(list1, list2))
        random.shuffle(zipped)
        list1, list2 = zip(*zipped)
        return list(list1), list(list2)

    
    def label_encode(self, sequence):
        """Apply one hot encoding"""

        encoded_sequence = self.lb.transform(list(sequence))

        return encoded_sequence  # numpy array


    def kmer_encode(self):
        pass

    def __getitem__(self, i):
        """Gets the ith sequence from the dataset"""
        batch = {
            "sequences": self.sequences[i],
            "labels":self.labels[i]
        }
        return batch


    def __len__(self):
        return len(self.labels)


# For testing can just run python dna_dataset.py
if __name__ == "__main__":
    DNADataset(ACCESSIBLE_FILE, NOT_ACCESSIBLE_FILE)
