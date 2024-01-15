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

