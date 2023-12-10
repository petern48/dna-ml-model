import torch
from torch import nn

# https://towardsdatascience.com/nlp-with-cnns-a6aa743bdc1e#:~:text=CNNs%20can%20be%20used%20for,important%20for%20any%20learning%20algorithm.
class CNNModel(torch.nn.Module):
    def __init__(self, kernel_size, embed_dim, num_filters1, num_filters2, pool_kernel_size, hidden_dense1,
                 hidden_dense2, dropout_rate):
        super().__init__()

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
        self.dropout_rate = dropout_rate


        self.Conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=self.num_filters1, kernel_size=self.kernel_size)  #in_channel=1, out_channels=128, kernel_size=2)
        
        self.pool = nn.MaxPool1d(self.pool_kernel_size)

        self.Conv2 = nn.Conv1d(self.num_filters1, self.num_filters2, self.kernel_size)

        self.flatten = nn.Flatten(start_dim=1)  # start flattening after 1st (BATCH_SIZE) dim


        # dense_input = batchsize * num_filters2 * 
        dense_input = 3136
        self.linear1 = nn.Linear(dense_input, self.hidden_dense1)
        self.linear2 = nn.Linear(self.hidden_dense1, self.hidden_dense2)

        self.linear3 = nn.Linear(self.hidden_dense2, 1)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.functional.relu
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, sequence_input):
        x = self.relu(self.Conv1(sequence_input))
        x = self.pool(x)

        x = self.relu(self.Conv2(x))
        x = self.pool(x)

        x = self.flatten(x)  #, start_dim=1)  # start flattening after BATCH_SIZE dim

        # print("After flatten", x.shape)

        x = self.linear1(x)
        x = self.dropout(x)

        x = self.linear2(x)

        x = self.linear3(x)

        return self.sigmoid(x)  # return value between 0 and 1


def save_CNNModel(model_save_path, model):
    checkpoint = {
        "embed_dim": model.embed_dim,
        "kernel_size": model.kernel_size,
        "num_filters1": model.num_filters1,
        "num_filters2": model.num_filters2,
        "pool_kernel_size": model.pool_kernel_size,
        "hidden_dense1": model.hidden_dense1,
        "hidden_dense2": model.hidden_dense2,
        "dropout_rate" : model.dropout_rate,

        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, model_save_path)


def load_CNNModel(model_save_path):
    checkpoint = torch.load(model_save_path)
    model = CNNModel(
        embed_dim=checkpoint["embed_dim"],
        kernel_size=checkpoint["kernel_size"],
        num_filters1=checkpoint["num_filters1"],
        num_filters2=checkpoint["num_filters2"],
        pool_kernel_size=checkpoint["pool_kernel_size"],
        hidden_dense1=checkpoint["hidden_dense1"],
        hidden_dense2=checkpoint["hidden_dense2"],
        dropout_rate=checkpoint["dropout_rate"]
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model


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
    # print(encoded.shape)
    model = CNNModel(embed_dim=4)
    # model(torch.Tensor(encoded))
    train_dataset = DNADataset(ACCESSIBLE_FILE, ACCESSIBLE_FILE)
    model = CNNModel(embed_dim=4)
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