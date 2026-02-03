import torch

# Vocabulary Settings
MAX_ATOMIC_NUM = 118
MASK_ATOM_ID = MAX_ATOMIC_NUM + 1          # 119
ATOM_VOCAB_SIZE = MAX_ATOMIC_NUM + 2       # 120  (0=pad, 1–118=elements, 119=mask)

MAX_BOND_TYPE = 4
MASK_BOND_ID = MAX_BOND_TYPE + 1           # 5
BOND_VOCAB_SIZE = MAX_BOND_TYPE + 2        # 6    (0=no-bond/pad, 1–4=bond types, 5=mask)

# Training Hyperparameters
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# Model Architecture
HIDDEN_DIM = 128
NUM_LAYERS = 5          # FIX #9: was 3, now matches train_GIN.py
NUM_HEADS = 12

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
