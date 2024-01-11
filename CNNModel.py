import torch
from torch import nn


# https://towardsdatascience.com/nlp-with-cnns-a6aa743bdc1e#:~:text=CNNs%20can%20be%20used%20for,important%20for%20any%20learning%20algorithm.
class CNNModel(torch.nn.Module):
    def __init__(self, kernel_size, embed_dim, conv_filters, pool_kernel_size, linear_neurons, dropout_rate_Dense, use_conv_dropout=False, out_size=1):
        super().__init__()

        self.seq_length = 200
        # output size (N-F)/S +1 where N size image, F size filter, S size stride
        # could use padding to get same size output
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.dropout_rate_Dense = dropout_rate_Dense

        # torch.nn.Embedding(num_embeddings, embedding_dim)

        self.conv_filters = conv_filters
        all_conv_filters = conv_filters.copy()
        all_conv_filters.insert(0, embed_dim)  # add input dim at beginning of con

        self.Convs = nn.ModuleList([nn.Conv1d(all_conv_filters[i-1],all_conv_filters[i],self.kernel_size, padding=1) for i in range(1, len(all_conv_filters))])
        # self.batch_norms
        self.relu = nn.functional.relu
        self.pool = nn.MaxPool1d(self.pool_kernel_size) # stride=self.pool_kernel_size)  # led to worse results
        self.flatten = nn.Flatten(start_dim=1)  # start flattening after 1st (BATCH_SIZE) dim

        linear_input = all_conv_filters[-1] * int(self.seq_length / (pool_kernel_size ** len(self.Convs)))

        self.linear_neurons = linear_neurons
        all_linear_neurons = linear_neurons.copy()  # Add the begin and end dims
        all_linear_neurons.insert(0, linear_input)
        all_linear_neurons.insert(len(all_linear_neurons), out_size)  # Add 1 to end

        self.linears = nn.ModuleList([nn.Linear(all_linear_neurons[i-1], all_linear_neurons[i]) for i in range(1, len(all_linear_neurons))])
        self.dropout_Dense = nn.Dropout(self.dropout_rate_Dense)
        self.use_conv_dropout = use_conv_dropout
        self.dropout_Conv = nn.Dropout(0.2)
        # self.batch_norm = nn.BatchNorm1d(128)  # num_filters1 i think
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, use_sigmoid=False):
        """x is sequence input"""
        for i in range(len(self.Convs)):
            x = self.Convs[i](x)
            if self.use_conv_dropout:
                x = self.dropout_Conv(x)
            x = self.relu(x)
            x = self.pool(x)
        # x = self.relu(self.batch_norm(x))   # before feeing into relu

        x = self.flatten(x)  #, start_dim=1)  # start flattening after BATCH_SIZE dim

        for i in range(len(self.linears)):
            x = self.linears[i](x)
            self.dropout_Dense(x)

        if use_sigmoid:
            x = self.sigmoid(x) # return value between 0 and 1

        return x


def save_CNNModel(model_save_path, model):
    checkpoint = {
        "kernel_size": model.kernel_size,
        "embed_dim": model.embed_dim,
        "conv_filters": model.conv_filters,
        "pool_kernel_size": model.pool_kernel_size,
        "linear_neurons": model.linear_neurons,
        "dropout_rate_Dense" : model.dropout_rate_Dense,
        'state_dict': model.state_dict()

    }
    torch.save(checkpoint, model_save_path)

def load_CNNModel(model_save_path):
    checkpoint = torch.load(model_save_path)
    # checkpoint["linear_neurons"].pop(len(checkpoint["linear_neurons"]) - 1)
    # checkpoint["linear_neurons"].pop(0)
    # checkpoint["conv_filters"].pop(0)
    model = CNNModel(
        embed_dim=checkpoint["embed_dim"],
        kernel_size=checkpoint["kernel_size"],
        conv_filters=checkpoint["conv_filters"],
        pool_kernel_size=checkpoint["pool_kernel_size"],
        linear_neurons=checkpoint["linear_neurons"],
        dropout_rate_Dense=checkpoint["dropout_rate_Dense"]
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model


# IGNORE THIS: this is for debugging/testing
if __name__ == "__main__":
    from dna_dataset import *
    import numpy as np
    BATCH_SIZE = 64
    sequence = "GAGACCCTTTGGTTAGCTTTCCACGCCAAGTGGCCGTTCCAGGCAGGCAGTGTCGTCTTGGTTCAGCCAAGGTCACAGAGGGAGTGATAGCTTCCGCGCAGCCCTGGCTACGGACTCTGGGCATCTTTCCACTGCCCCGCTTGCGCCACCTGTTAGGCAGGATCGTTTTTCCTCTGGGGCAAGATCAAAATCCAGGTCCT"
    # print("length sequence", len(sequence))
    bases = ["A", "C", "G", "T"]
    lb = LabelBinarizer()
    lb.fit_transform(bases)
    encoded = lb.transform(list(sequence))
    encoded = np.transpose(encoded)
    encoded = np.tile(encoded, (BATCH_SIZE, 1, 1))  # stack batch_size copies of encoding together
    encoded = torch.Tensor(encoded)
    # print(encoded.shape)
    conv_filters = [128,64,32]
    linear_neurons = [64, 32]
    conv_filters.insert(0, 4)  # add input dim at beginning of con
    conv_filters = conv_filters
    pool_kernel_size = 2
    Convs = nn.ModuleList([nn.Conv1d(conv_filters[i-1],conv_filters[i],3, padding=1) for i in range(1, len(conv_filters))])
    relu = nn.functional.relu
    pool = nn.MaxPool1d(pool_kernel_size) # stride=self.pool_kernel_size)  # led to worse results
    flatten = nn.Flatten(start_dim=1)  # start flattening after 1st (BATCH_SIZE) dim
    # dense_input = batchsize * num_filters2 *
    linear_input = conv_filters[-1] * int(200 / (pool_kernel_size ** len(Convs)))

    linear_neurons.insert(0, linear_input)
    linear_neurons.insert(len(linear_neurons), 1)  # Add 1 to end

    linears = nn.ModuleList([nn.Linear(linear_neurons[i-1], linear_neurons[i]) for i in range(1, len(linear_neurons))])
    print(encoded.shape)
    for i in range(len(Convs)):
        encoded = Convs[i](encoded)
        encoded = relu(encoded)
        encoded = pool(encoded)
        print(encoded.shape)
    encoded = flatten(encoded)
    print("lin input", linear_input)
    print(encoded.shape)
    for i in range(len(Convs)):
        encoded = linears[i](encoded)
        print(encoded.shape)


# https://medium.com/analytics-vidhya/predicting-genes-with-cnn-bdf278504e79
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=21, input_shape=(train_features.shape[1], 4), 
#                  padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=200, kernel_size=21, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))

# epochs = 50
# lrate = 0.01
# decay = lrate / epochs
# sgd = SGD(lr = lrate, momentum = 0.90, decay = decay, nesterov = False)
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])
# model.summary()