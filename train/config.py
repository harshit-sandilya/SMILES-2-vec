import torch

# ---------------- Vocabulary ----------------
MAX_ATOMIC_NUM = 118
MASK_ATOM_ID = MAX_ATOMIC_NUM + 1
ATOM_VOCAB_SIZE = MAX_ATOMIC_NUM + 2

MAX_BOND_TYPE = 4
MASK_BOND_ID = MAX_BOND_TYPE + 1
BOND_VOCAB_SIZE = MAX_BOND_TYPE + 2


# ---------------- Training ----------------
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
VAL_CHECK_STEPS = 100000
WARMUP_STEPS = 2000


# ---------------- Model ----------------
HIDDEN_DIM = 768
NUM_LAYERS = 8
NUM_HEADS = 12
EMBEDDING_DIM = 256
PROPERTY_LOSS_WEIGHT = 0.5


# ---------------- Hardware ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 32
PIN_MEMORY = True
PERSISTENT_WORKERS = True
