import os

from lightning.data import StreamingDataset
from torch_geometric.data import Data

input_dir = "optimized_graph_dataset"

if not os.path.exists(input_dir):
    print(f"Error: The directory '{input_dir}' does not exist.")
    print("Please run your updated 'optimise_dataset.py' script first.")
else:
    print(f"Loading dataset from '{input_dir}'...")

    dataset = StreamingDataset(input_dir=input_dir)

    total_items = len(dataset)
    print(f"Successfully loaded dataset with {total_items:,} items.")

    if total_items == 0:
        print(
            "The dataset is empty. Check the 'optimise_dataset.py' script for potential issues."
        )
    else:
        num_to_print = min(5, total_items)
        print(f"\nPrinting the first {num_to_print} entries:\n")

        for i in range(num_to_print):
            item = dataset[i]

            print(f"--- Item {i} ---")

            print(f"Type of loaded item: {type(item)}")

            if isinstance(item, Data):
                print("\n  1. Atomic Numbers Tensor:")
                print(f"     - Shape: {item.x.shape}")
                print(f"     - Dtype: {item.x.dtype}")
                print(f"     - Data (first 15): {item.x[:15]}...")

                print("\n  2. Bond Matrix Tensor:")
                print(f"     - Shape: {item.edge_index.shape}")
                print(f"     - Dtype: {item.edge_index.dtype}")

                print("\n  3. Edge Attributes Tensor:")
                print(f"     - Shape: {item.edge_attr.shape}")
                print(f"     - Dtype: {item.edge_attr.dtype}")

                print("\n  4. Target Tensors:")
                print(f"     - Shape: {item.y_atoms.shape}, {item.y_bonds.shape}")
                print(f"     - Dtype: {item.y_atoms.dtype}, {item.y_bonds.dtype}")
            else:
                print(
                    f"  - Unexpected item format. Expected a Data object, but got: {item}"
                )

            print("-" * (len(f"--- Item {i} ---")) + "\n")
