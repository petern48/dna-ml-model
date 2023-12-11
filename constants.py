LINES_PER_SEQUENCE = 4
ACCESSIBLE_LABEL = 1
NOT_ACCESSIBLE_LABEL = 0
DATA_ZIP_FILE = "Files.zip"
DATA_DIR = DATA_ZIP_FILE.strip(".zip")
ACCESSIBLE_FILE = f"{DATA_DIR}/accessible.fasta"
# NOT_ACCESSIBLE_FILE = f"{DATA_DIR}/notaccessible.fasta"
NOT_ACCESSIBLE_FILE = f"{DATA_DIR}/reduced_nonaccessible.fasta"
TEST_FILE = f"{DATA_DIR}/test.fasta"
VALIDATION_SPLIT = 0.15
PRETRAINED_DIR = "pretrained"
SOLUTION_FILE = "predictions.csv"


# TRAINING LOOP
EPOCHS = 100
BATCH_SIZE = 100  # 64
N_EVAL = 600
LEARNING_RATE = 0.01  # OR .0005  # originally 0.05
# loss_fn and optimizer can be modified in main.ipynb
