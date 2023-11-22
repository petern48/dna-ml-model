import torch
import random
from constants import *

class DNADataset(torch.utils.data.Dataset):
    def __init__(self, acc_data_path, not_acc_data_path):
        random.seed(1)  # for consistent results
        self.accessible_count = 0
        self.not_accessible_count = 0
        self.sequences = []
        self.labels = []

        self.accessible_count = self.read_data_file(acc_data_path, self.sequences, self.labels, accessible=True)
        self.not_accessible_count = self.read_data_file(not_acc_data_path, self.sequences, self.labels, accessible=False)

        assert(self.accessible_count + self.not_accessible_count == len(self.sequences))

        self.sequences, self.labels = self.shuffle_lists(self.sequences, self.labels)


    def read_data_file(self, data_file, sequence_list, label_list, accessible=True):

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

                # TODO: UNSURE if should collapse to one sequence or turn into matrix for cnn
                for _ in range(LINES_PER_SEQUENCE):  # read the 4 lines of dna sequence
                    sequence += next(f)

                # append the sequence and label
                sequence_list.append(sequence.rstrip())
                label_list.append(label)
        
        return seq_count

    
    def shuffle_lists(self, list1, list2):
        zipped = list(zip(list1, list2))
        random.shuffle(zipped)
        list1, list2 = zip(*zipped)
        return list(list1), list(list2)


    def __getitem__(self, i):
        """Gets the ith sequence from the dataset"""
        batch = {
            "sequence": self.sequences[i],
            "label":self.labels[i]
        }
        return batch


    def __len__(self):
        return len(self.labels)
