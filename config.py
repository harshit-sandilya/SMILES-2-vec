import torch

MAX_ATOMIC_NUM = 118
MASK_ATOM_ID = MAX_ATOMIC_NUM + 1
ATOM_VOCAB_SIZE = MAX_ATOMIC_NUM + 2

MAX_BOND_TYPE = 4
MASK_BOND_ID = MAX_BOND_TYPE + 1
BOND_VOCAB_SIZE = MAX_BOND_TYPE + 2

EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_dim = 512
num_layers = 8
num_heads = 12
