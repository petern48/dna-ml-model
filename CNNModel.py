import torch
from torch import nn


# https://towardsdatascience.com/nlp-with-cnns-a6aa743bdc1e#:~:text=CNNs%20can%20be%20used%20for,important%20for%20any%20learning%20algorithm.
class CNNModel(torch.nn.Module):
    def __init__(self, kernel_size, embed_dim, conv_filters, pool_kernel_size, linear_neurons, dropout_rate_Dense):
        super().__init__()

        self.seq_length = 200
        # output size (N-F)/S +1 where N size image, F size filter, S size stride
        # could use padding to get same size output
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        # self.num_filters1 = num_filters1
        # self.num_filters2 = num_filters2
        self.pool_kernel_size = pool_kernel_size
        # self.hidden_dense1 = hidden_dense1
        # self.hidden_dense2 = hidden_dense2
        self.dropout_rate_Dense = dropout_rate_Dense

        # torch.nn.Embedding(num_embeddings, embedding_dim)

        conv_filters.insert(0, embed_dim)  # add input dim at beginning of con
        self.conv_filters = conv_filters

        self.Convs = nn.ModuleList([nn.Conv1d(conv_filters[i-1],conv_filters[i],self.kernel_size, padding=1) for i in range(1, len(conv_filters))])
        self.relu = nn.functional.relu
        self.pool = nn.MaxPool1d(self.pool_kernel_size) # stride=self.pool_kernel_size)  # led to worse results
        # self.Conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=self.num_filters1, kernel_size=self.kernel_size, padding=1)  #in_channel=1, out_channels=128, kernel_size=2)
        # if num_filters2 != 0:
        #     self.Conv2 = nn.Conv1d(self.num_filters1, self.num_filters2, self.kernel_size, padding=1)
        #     num_out = num_filters2
        # else:
        #     num_out = num_filters1
        self.flatten = nn.Flatten(start_dim=1)  # start flattening after 1st (BATCH_SIZE) dim

        # dense_input = batchsize * num_filters2 * 
        linear_input = conv_filters[-1] * int(self.seq_length / (2 * self.pool_kernel_size))

        linear_neurons.insert(0, linear_input)
        linear_neurons.insert(len(linear_neurons), 1)  # Add 1 to end
        self.linear_neurons = linear_neurons

        self.linears = nn.ModuleList([nn.Linear(linear_neurons[i-1], linear_neurons[i]) for i in range(1, len(linear_neurons))])
        self.dropout_Dense = nn.Dropout(self.dropout_rate_Dense)

        # self.linear1 = nn.Linear(dense_input, self.hidden_dense1)
        # if hidden_dense2 != 0:
        #     self.linear2 = nn.Linear(self.hidden_dense1, self.hidden_dense2)
        #     dense3_input = hidden_dense2
        # else:
        #     dense3_input = hidden_dense1
        # self.linear3 = nn.Linear(dense3_input, 1)

        # self.dropout_Conv = nn.Dropout(self.dropout_rate_Conv)
        # self.batch_norm = nn.BatchNorm1d(128)  # num_filters1 i think
        # self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        """x is sequence input"""
        for i in range(len(self.Convs)):
            x = self.Convs[i](x)
            x = self.relu(x)
            x = self.pool(x)
        # x = self.relu(self.Conv1(x))

        # x = self.relu(self.batch_norm(x))   # before feeing into relu
        # if self.num_filters2 != 0:
        #     x = self.relu(self.Conv2(x))
        #     x = self.pool(x)

        x = self.flatten(x)  #, start_dim=1)  # start flattening after BATCH_SIZE dim

        # print("After flatten", x.shape)

        for i in range(len(self.linears)):
            x = self.linears[i](x)
            self.dropout_Dense(x)
        # x = self.linear1(x)
        # x = self.dropout_Dense(x)
        # if self.linear2 != 0:
        #     x = self.linear2(x)
        #     x = self.dropout_Dense(x)
        # x = self.linear3(x)
        # x = self.dropout_Dense(x)

        return x
        # return self.sigmoid(x)  # return value between 0 and 1


def save_CNNModel(model_save_path, model):
    checkpoint = {
        "kernel_size": model.kernel_size,
        "embed_dim": model.embed_dim,
        "conv_filters": model.conv_filters,
        "pool_kernel_size": model.pool_kernel_size,
        "linear_neurons": model.linear_neurons,
        "dropout_rate_Dense" : model.dropout_rate_Dense,
        'state_dict': model.state_dict()
        # "num_filters1": model.num_filters1,
        # "num_filters2": model.num_filters2,
        # "hidden_dense1": model.hidden_dense1,
        # "hidden_dense2": model.hidden_dense2,

    }
    torch.save(checkpoint, model_save_path)

def load_CNNModel(model_save_path):
    checkpoint = torch.load(model_save_path)
    checkpoint["linear_neurons"].pop(len(checkpoint["linear_neurons"]) - 1)
    checkpoint["linear_neurons"].pop(0)
    checkpoint["conv_filters"].pop(0)
    model = CNNModel(
        embed_dim=checkpoint["embed_dim"],
        kernel_size=checkpoint["kernel_size"],
        conv_filters=checkpoint["conv_filters"],
        pool_kernel_size=checkpoint["pool_kernel_size"],
        linear_neurons=checkpoint["linear_neurons"],
        dropout_rate_Dense=checkpoint["dropout_rate_Dense"]
        # num_filters1=checkpoint["num_filters1"],
        # num_filters2=checkpoint["num_filters2"],
        # hidden_dense1=checkpoint["hidden_dense1"],
        # hidden_dense2=checkpoint["hidden_dense2"],
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
    model = CNNModel(
                    kernel_size=2,
                    embed_dim=4,
                    num_filters1=128,
                    num_filters2=64,
                    pool_kernel_size=2,
                    hidden_dense1=128,
                    hidden_dense2=64,
                    dropout_rate_Dense=0.5
    )
    x = model(encoded)
    print(x.shape)
    raise
    train_dataset = DNADataset(ACCESSIBLE_FILE, ACCESSIBLE_FILE)
    # model = CNNModel(embed_dim=4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    for batch in train_loader:
        output = model(batch["sequence"])

        print(output)
        sys.exit()

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