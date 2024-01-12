import torch
from torch import nn
from CNNModel import CNNModel


SPLIT_DIM = 1

class HybridModel(torch.nn.Module):
    def __init__(self, kernel_size=3, embed_dim=4, conv_filters=[128, 128, 128],
                 pool_kernel_size=2, linear_neurons=[256], dropout_rate_Dense=0.5,
                 cnn_out_size=128, ffn_out_size=128, k=6):
        super().__init__()
        # input size: (1, 4, 1224)  1024 + 200
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.conv_filters = conv_filters
        self.pool_kernel_size = pool_kernel_size
        self.linear_neurons = linear_neurons
        self.dropout_rate_Dense = dropout_rate_Dense
        self.cnn_out_size = cnn_out_size
        self.ffn_out_size = ffn_out_size
        self.k = k

        self.cnn = CNNModel(kernel_size,
                    embed_dim,
                    conv_filters, 
                    pool_kernel_size,
                    linear_neurons,
                    dropout_rate_Dense,
                    out_size=cnn_out_size)
        self.flatten = nn.Flatten(start_dim=1)
        self.ffn = FFN(input_size=(4**k), out_size=ffn_out_size)  # 4**k permutaions
        self.dropout = nn.Dropout(dropout_rate_Dense)

        # merge_size = 256
        # self.linear = nn.Linear(merge_size, 256)
        self.linear_final = nn.Linear(cnn_out_size + ffn_out_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, use_sigmoid=True):
        # Slice   # torch.split(x, dim=3)  # unsure about dim
        a, b = x[:,:,:200], x[:,:,200:]
        # assert(SPLIT_DIM == 1)  # if this fails, change above splicing
        # print(f"a {a.shape}")
        # print(f"b {b.shape}")
        # a = a.transpose(1, 2)
        a = self.cnn(a)
        b = self.flatten(b)
        b = self.ffn(b)

        # print(f"a {a.shape}")
        # print(f"b {b.shape}")
        # a = a.transpose(1, 2)  # transpose back

        merged_value = torch.cat((a, b), dim=1)  # unsure of dim
        # print("merged", merged_value.shape)
        x = self.dropout(merged_value)
        # x = self.linear(x)
        x = self.linear_final(x)
        if use_sigmoid:
            x = self.sigmoid(x)
        return x

def save_HybridModel(model_save_path, model):
    # checkpoint
    ckpt = {
        "kernel_size":model.kernel_size,
        "embed_dim":model.embed_dim,
        "conv_filters": model.conv_filters,
        "pool_kernel_size":model.pool_kernel_size,
        "linear_neurons": model.linear_neurons,
        "dropout_rate_Dense":model.dropout_rate_Dense,
        "cnn_out_size":model.cnn_out_size,
        "ffn_out_size": model.ffn_out_size,
        "k": model.k,
        "state_dict": model.state_dict()
    }
    torch.save(ckpt, model_save_path)


def load_HybridModel(model_save_path):
    ckpt = torch.load(model_save_path)
    model = HybridModel(
            kernel_size=ckpt["kernel_size"],
            embed_dim=ckpt["embed_dim"],
            conv_filters=ckpt["conv_filters"],
            pool_kernel_size=ckpt["pool_kernel_size"],
            linear_neurons=ckpt["linear_neurons"],
            dropout_rate_Dense=ckpt["dropout_rate_Dense"],
            cnn_out_size=ckpt["cnn_out_size"],
            ffn_out_size=ckpt["ffn_out_size"],
            k=ckpt["k"]
    )
    model.load_state_dict(ckpt['state_dict'])
    return model


class FFN(torch.nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        # input to FFN is (-1, 4096, 1)
        self.linear = nn.Linear(input_size, out_size)
        # self.dropout_Dense = nn.Dropout(self.dropout_rate_Dense)
    def forward(self, x):
        return self.linear(x)


# NOT BEING USED
class CNN(torch.nn.Module):
    def __init__(self, embed_dim, num_layers=3):
        super().__init__()
        # input of (-1, 4, 200)
        self.seq_length = 200
        pool_size = 5
        # output size (N-F)/S +1 where N size image, F size filter, S size stride
        # could use padding to get same size output
        self.embed_dim = embed_dim
        Convs = []
        for _ in range(num_layers):
            Convs.append(nn.Conv2d(embed_dim, 128, kernel_size=(4,13)))
            Convs.append(nn.Conv2d(128, 128, kernel_size=(1,7)))
            Convs.append(nn.Conv2d(128, 128, kernel_size=(1,5)))
        self.Convs = nn.ModuleList(Convs)
        assert(len(self.Convs) == 3 * num_layers)
        # Make 9 actual copies
        self.relu = nn.functional.relu
        self.flatten = nn.Flatten(start_dim=1)  # start flattening after 1st (BATCH_SIZE) dim
        self.pool = nn.MaxPool1d(pool_size) # stride=self.pool_kernel_size)  # led to worse results
        linear_input = 5
        self.linear = nn.Linear(linear_input, 256)  # TODO
        # self.batch_norm = nn.BatchNorm1d(128)  # num_filters1 i think
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """x is sequence input"""
        for i in range(len(self.Convs)):
            print(i)
            x = self.relu(self.Convs[i](x))
            if i + 1 % 3 == 0:
                x = self.pool(x)
        # x = self.relu(self.batch_norm(x))   # before feeing into relu
        x = self.flatten(x, start_dim=1)  # start flattening after BATCH_SIZE dim
        x = self.linear(x)
        return x


# Not being used
def seq_to_mat(seq):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':4, 'N':4}
    mat = np.zeros((len(seq),5))  
    for i in range(len(seq)):
        mat[i,encoding_matrix[seq[i]]] = 1
    mat = mat[:,:4]
    return mat
# [0. 1. 0. 0.]
# [0. 0. 0. 1.]]
# (200,4)


def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        kspec_vec[index] += 1
    return kspec_vec
# (4096, 1)


# IGNORE THIS: this is for debugging/testing
from sklearn.preprocessing import LabelBinarizer
bases = ["A", "C", "G", "T"]
lb = LabelBinarizer()
lb.fit_transform(bases)
def label_binarize(seq):
    return lb.transform(list(seq))

if __name__ == "__main__":
    from dna_dataset import *
    import numpy as np
    BATCH_SIZE = 64
    sequence = "GAGACCCTTTGGTTAGCTTTCCACGCCAAGTGGCCGTTCCAGGCAGGCAGTGTCGTCTTGGTTCAGCCAAGGTCACAGAGGGAGTGATAGCTTCCGCGCAGCCCTGGCTACGGACTCTGGGCATCTTTCCACTGCCCCGCTTGCGCCACCTGTTAGGCAGGATCGTTTTTCCTCTGGGGCAAGATCAAAATCCAGGTCCT"
    embed_dim = 4
    # encoded = seq_to_mat(sequence)
    kmers = seq_to_kspec(sequence)
    one_hot = np.array(label_binarize(sequence), dtype=np.float64)
    # kmers = np.expand_dims(kmers, axis=0)
    # one_hot = np.expand_dims(one_hot, axis=0)
    # one_hot = np.expand_dims(one_hot, axis=0)  # again
    # one_hot = one_hot.transpose(0, 1, 3, 2)
    one_hot = one_hot.transpose(1, 0)
    kmers = kmers.reshape(embed_dim, int(kmers.shape[0] / embed_dim))
    kmers = torch.Tensor(kmers)
    one_hot = torch.Tensor(one_hot).float()
    print(f"kmers {kmers.shape}")
    print(f"one_hot {one_hot.shape}")

    x = torch.cat((one_hot, kmers), dim=SPLIT_DIM)
    print("concatenated input", x.shape)

    x = x.unsqueeze(dim=0)  # to mimic batch size
    print("unsqueezed shape", x.shape)

    # a, b = x[:,:,:200], x[:, :,200:]
    # assert(SPLIT_DIM == 1)  # if this fails, change above splicing
    # print(f"a {a.shape}")
    # print(f"b {b.shape}")

    # cnn = CNNModel(kernel_size=3,
    #         embed_dim=embed_dim,
    #         conv_filters=[128, 128, 128], 
    #         pool_kernel_size=2,
    #         linear_neurons=[256],
    #         dropout_rate_Dense=0.5,
    #         use_conv_dropout=False)
    # cnn(a)

    # one_hot = one_hot[:,:,0:4,:]
    # print("input_shape", one_hot.shape)
    # out = cnn(one_hot)
    model = HybridModel(embed_dim=4)
    x = model(x)
    print(x.shape)

    # flatten = nn.Flatten(start_dim=1)
    # kmers = flatten(kmers)
    # print(kmers.shape)
    # fnn = FNN(input_size=4096)
    # output = fnn(kmers)
