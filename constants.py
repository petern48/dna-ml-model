LINES_PER_SEQUENCE = 4
ACCESSIBLE_LABEL = 1
NOT_ACCESSIBLE_LABEL = 0
DATA_ZIP_FILE = "Files.zip"
DATA_DIR = DATA_ZIP_FILE.strip(".zip")
ACCESSSIBLE_FILE = f"{DATA_DIR}/accessible.fasta"
NOT_ACCESSIBLE_FILE = f"{DATA_DIR}/notaccessible.fasta"
TEST_FILE = f"{DATA_DIR}/test.fasta"


# TRAINING LOOP
EPOCHS = 100
BATCH_SIZE = 32
N_EVAL = 100
# loss_fn and optimizer can be modified in main.ipynb



# CNN Model
# embedding_dims = 300 #Length of the token vectors
filters = 250 #number of filters in your Convnet
kernel_size = 3 # a window size of 3 tokens
hidden_dims = 250 #number of neurons at the normal feedforward NN

