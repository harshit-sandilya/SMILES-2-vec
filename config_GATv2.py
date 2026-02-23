import torch

# Vocabulary Settings
MAX_ATOMIC_NUM = 118
MASK_ATOM_ID = MAX_ATOMIC_NUM + 1          # 119
ATOM_VOCAB_SIZE = MAX_ATOMIC_NUM + 2       # 120  (0=pad, 1–118=elements, 119=mask)

MAX_BOND_TYPE = 4
MASK_BOND_ID = MAX_BOND_TYPE + 1           # 5
BOND_VOCAB_SIZE = MAX_BOND_TYPE + 2        # 6    (0=no-bond/pad, 1–4=bond types, 5=mask)

# Training Hyperparameters
EPOCHS = 60                  # Increased from 30 - larger models need more epochs
BATCH_SIZE = 32              # Increased from 24 - better GPU utilization
LEARNING_RATE = 2e-4         # Slightly higher from 1e-4 - faster convergence

# Model Architecture - SMART SCALING FOR GATV2
HIDDEN_DIM = 512             # Increased from 128 → 5.39M params
NUM_LAYERS = 6               # Keep at 6 for good depth
NUM_HEADS = 8                # Changed from 12 → Each head gets 64 dims (512/8)

# Contrastive Learning (Optional but recommended)
USE_CONTRASTIVE_LOSS = False      # Start without it, enable if needed
CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_WEIGHT = 0.1

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"