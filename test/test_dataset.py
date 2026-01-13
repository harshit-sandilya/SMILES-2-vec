# import os

# from lightning.data import StreamingDataset
# from torch_geometric.data import Data
# BASE_DATA_DIR = "data"
# BASE_RESULTS_DIR = "results"

# input_dir = os.path.join(BASE_DATA_DIR, "optimized_graph_dataset")
# os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
# report_lines = []

# if not os.path.exists(input_dir):
#     print(f"Error: The directory '{input_dir}' does not exist.")
#     print("Please run your updated 'optimise_dataset.py' script first.")
# else:
#     print(f"Loading dataset from '{input_dir}'...")

#     dataset = StreamingDataset(input_dir=input_dir)

#     total_items = len(dataset)
#     print(f"Successfully loaded dataset with {total_items:,} items.")

#     if total_items == 0:
#         print(
#             "The dataset is empty. Check the 'optimise_dataset.py' script for potential issues."
#         )
#     else:
#         num_to_print = min(5, total_items)
#         print(f"\nPrinting the first {num_to_print} entries:\n")

#         for i in range(num_to_print):
#             item = dataset[i]

#             print(f"--- Item {i} ---")

#             print(f"Type of loaded item: {type(item)}")

#             if isinstance(item, Data):
#                 print("\n  1. Atomic Numbers Tensor:")
#                 print(f"     - Shape: {item.x.shape}")
#                 print(f"     - Dtype: {item.x.dtype}")
#                 print(f"     - Data (first 15): {item.x[:15]}...")

#                 print("\n  2. Bond Matrix Tensor:")
#                 print(f"     - Shape: {item.edge_index.shape}")
#                 print(f"     - Dtype: {item.edge_index.dtype}")

#                 print("\n  3. Edge Attributes Tensor:")
#                 print(f"     - Shape: {item.edge_attr.shape}")
#                 print(f"     - Dtype: {item.edge_attr.dtype}")

#                 print("\n  4. Target Tensors:")
#                 print(f"     - Shape: {item.y_atoms.shape}, {item.y_bonds.shape}")
#                 print(f"     - Dtype: {item.y_atoms.dtype}, {item.y_bonds.dtype}")
#             else:
#                 print(
#                     f"  - Unexpected item format. Expected a Data object, but got: {item}"
#                 )

#             print("-" * (len(f"--- Item {i} ---")) + "\n")



import os

from lightning.data import StreamingDataset
from torch_geometric.data import Data

# ===================== Base Directories =====================
BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"

# Dataset location (DATA folder)
input_dir = os.path.join(BASE_DATA_DIR, "optimized_graph_dataset")

# Ensure results directory exists
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Collect report lines to save in results/
report_lines = []

if not os.path.exists(input_dir):
    msg = f"Error: The directory '{input_dir}' does not exist."
    print(msg)
    report_lines.append(msg)

    msg = "Please run your updated 'optimise_dataset.py' script first."
    print(msg)
    report_lines.append(msg)
else:
    msg = f"Loading dataset from '{input_dir}'..."
    print(msg)
    report_lines.append(msg)

    dataset = StreamingDataset(input_dir=input_dir)

    total_items = len(dataset)
    msg = f"Successfully loaded dataset with {total_items:,} items."
    print(msg)
    report_lines.append(msg)

    if total_items == 0:
        msg = (
            "The dataset is empty. Check the 'optimise_dataset.py' script for potential issues."
        )
        print(msg)
        report_lines.append(msg)
    else:
        num_to_print = min(5, total_items)
        msg = f"\nPrinting the first {num_to_print} entries:\n"
        print(msg)
        report_lines.append(msg)

        for i in range(num_to_print):
            item = dataset[i]

            header = f"--- Item {i} ---"
            print(header)
            report_lines.append(header)

            msg = f"Type of loaded item: {type(item)}"
            print(msg)
            report_lines.append(msg)

            if isinstance(item, Data):
                msg = f"Atomic Numbers: shape={item.x.shape}, dtype={item.x.dtype}"
                print(msg)
                report_lines.append(msg)

                msg = f"Edge Index: shape={item.edge_index.shape}, dtype={item.edge_index.dtype}"
                print(msg)
                report_lines.append(msg)

                msg = f"Edge Attr: shape={item.edge_attr.shape}, dtype={item.edge_attr.dtype}"
                print(msg)
                report_lines.append(msg)

                msg = (
                    f"Targets: y_atoms shape={item.y_atoms.shape}, "
                    f"y_bonds shape={item.y_bonds.shape}"
                )
                print(msg)
                report_lines.append(msg)
            else:
                msg = (
                    "Unexpected item format. Expected a torch_geometric.data.Data object."
                )
                print(msg)
                report_lines.append(msg)

            separator = "-" * len(header)
            print(separator + "\n")
            report_lines.append(separator + "\n")

# ===================== Save inspection report =====================
report_path = os.path.join(
    BASE_RESULTS_DIR, "optimized_dataset_inspection.txt"
)

with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"\n Inspection report saved to: {report_path}")
