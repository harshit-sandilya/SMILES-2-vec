from torch.utils.data import Dataset

from utils import create_masked_graph_from_tensors


class MaskedMoleculeDataset(Dataset):
    def __init__(self, tokenized_list, mask_ratio_atoms=0.15, mask_ratio_bonds=0.15):
        """
        Args:
            tokenized_list (list): A list of tokenized molecule dictionaries from SMILESTokenizer.
        """
        self.tokenized_list = tokenized_list
        self.mask_ratio_atoms = mask_ratio_atoms
        self.mask_ratio_bonds = mask_ratio_bonds

    def __len__(self):
        return len(self.tokenized_list)

    def __getitem__(self, idx):
        return create_masked_graph_from_tensors(
            self.tokenized_list[idx]["atomic_numbers"],
            self.tokenized_list[idx]["bond_matrix"],
            self.mask_ratio_atoms,
            self.mask_ratio_bonds,
        )
