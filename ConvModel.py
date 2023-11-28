import torch

# https://towardsdatascience.com/nlp-with-cnns-a6aa743bdc1e#:~:text=CNNs%20can%20be%20used%20for,important%20for%20any%20learning%20algorithm.
class ConvModel(torch.nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()

        # maybe use 1d
        # output size (N-F)/S +1 where N size image, F size filter, S size stride
        # could use padding to get same size output
        self.Conv1 = torch.nn.Conv1d() #in_channel=1, out_channels=128, kernel_size=2)
        self.pool1 = torch.nn.MaxPool2d()

        self.Conv2

        self.Conv3

        self.flatten = torch.flatten

        self.linear = torch.nn.Linear()

        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.nn.sigmoid()

    
    def forward(self, sequence_input):
        x = self.Conv1(sequence_input)
        # relu

        x = self.pool1(x)

        x = self.flatten(x)

        self.relu(x)


        return self.sigmoid(x)  # return value between 0 and 1


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