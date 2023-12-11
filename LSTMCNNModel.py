import torch
from torch import nn
from CNNModel import CNNModel

# https://towardsdatascience.com/nlp-with-cnns-a6aa743bdc1e#:~:text=CNNs%20can%20be%20used%20for,important%20for%20any%20learning%20algorithm.
class LSTMCNNModel(torch.nn.Module):
    def __init__(self, kernel_size, embed_dim, num_filters1, num_filters2, pool_kernel_size, hidden_dense1,
                 hidden_dense2, dropout_rate_Dense, lstm_units):
        super().__init__()

        self.seq_length = 200

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters1, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=2),
            nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=2)
        )

        self.lstm = nn.LSTM(embed_dim, lstm_units, batch_first=True)
        # Inputs: input, h_0, c_0
        # input shape (batchsize, sequence length, input_size)
        # h_0 shape (1or2 * num_layers, batchsize, hidden size)  # 1or2 for bidirectional or not
        # c_0 shape (1or2 * num_layers, batchsize, hidden size)
        dense_input_size = int(self.seq_length / (2 * pool_kernel_size))
        self.linear = nn.Linear(dense_input_size, 1)
        self.dropout_Dense = nn.Dropout(dropout_rate_Dense)

        self.sigmoid = nn.Sigmoid()
        # maybe use 1d
        # output size (N-F)/S +1 where N size image, F size filter, S size stride
        # could use padding to get same size output
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2

        self.pool_kernel_size = pool_kernel_size

        self.hidden_dense1 = hidden_dense1
        self.hidden_dense2 = hidden_dense2
        self.dropout_rate_Dense = dropout_rate_Dense

    
    def forward(self, x):
        """sequence_input"""
        # cnn input shape (batch_size, channels/embed dims, seq length)
        print(x.shape)
        x = self.cnn(x)
        print(x.shape)
        x = x.permute(0, 2, 1)
        print(x.shape)
        assert not torch.isnan(x).any()
        # permute to (batch_size, seq_len, input_size)
        x, (hn, cn) = self.lstm(x)
        assert not torch.isnan(x).any()
        print(x.shape)
        x = x.flatten(start_dim=1)
        print(x.shape)
        x = self.linear(x)
        print(x.shape)
        x = self.dropout_Dense(x)

        return self.sigmoid(x)


# def save_CNNModel(model_save_path, model):
#     checkpoint = {
#         "kernel_size": model.kernel_size,
#         "embed_dim": model.embed_dim,
#         "num_filters1": model.num_filters1,
#         "num_filters2": model.num_filters2,
#         "pool_kernel_size": model.pool_kernel_size,
#         "hidden_dense1": model.hidden_dense1,
#         "hidden_dense2": model.hidden_dense2,
#         "dropout_rate_Dense" : model.dropout_rate_Dense,

#         'state_dict': model.state_dict()
#     }
#     torch.save(checkpoint, model_save_path)

# def load_CNNModel(model_save_path):
#     checkpoint = torch.load(model_save_path)
#     model = CNNModel(
#         embed_dim=checkpoint["embed_dim"],
#         kernel_size=checkpoint["kernel_size"],
#         num_filters1=checkpoint["num_filters1"],
#         num_filters2=checkpoint["num_filters2"],
#         pool_kernel_size=checkpoint["pool_kernel_size"],
#         hidden_dense1=checkpoint["hidden_dense1"],
#         hidden_dense2=checkpoint["hidden_dense2"],
#         dropout_rate_Dense=checkpoint["dropout_rate_Dense"]
#     )
#     model.load_state_dict(checkpoint['state_dict'])
#     return model


# IGNORE THIS: this is for debugging/testing
if __name__ == "__main__":
    from dna_dataset import *
    import numpy as np
    sequence = "GAGACCCTTTGGTTAGCTTTCCACGCCAAGTGGCCGTTCCAGGCAGGCAGTGTCGTCTTGGTTCAGCCAAGGTCACAGAGGGAGTGATAGCTTCCGCGCAGCCCTGGCTACGGACTCTGGGCATCTTTCCACTGCCCCGCTTGCGCCACCTGTTAGGCAGGATCGTTTTTCCTCTGGGGCAAGATCAAAATCCAGGTCCT"
    # print("length sequence", len(sequence))
    bases = ["A", "C", "G", "T"]
    lb = LabelBinarizer()
    lb.fit_transform(bases)
    encoded = lb.transform(list(sequence))
    encoded = np.transpose(encoded)
    encoded = np.tile(encoded, (BATCH_SIZE, 1, 1))  # stack batch_size copies of encoding together
    encoded = torch.Tensor(encoded)
    print("input shape", encoded.shape)
    model = LSTMCNNModel(
                    kernel_size=2,
                    embed_dim=4,
                    num_filters1=128,
                    num_filters2=64,
                    pool_kernel_size=2,
                    hidden_dense1=128,
                    hidden_dense2=64,
                    dropout_rate_Dense=0.5,
                    lstm_units=1
    )
    x = model(encoded)
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