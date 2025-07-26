# SMILES Graph Neural Network Project

This project implements a Graph Neural Network (GNN) pipeline for molecular property prediction and representation learning using SMILES strings. The workflow includes data download, preprocessing, model training, embedding generation, visualization, and benchmarking against standard datasets.

## Prerequisites

- Python 3.10 or higher
- Required Python packages (see `requirements.txt`)
- Access to a GPU for training (optional but recommended)

## Setup Instructions

### Clone the repository:

```bash
git clone https://github.com/harshit-sandilya/SMILES-2-vec.git
cd SMILES-2-vec
```

### Install the required packages:

```bash
pip install -r requirements.txt
```

### Download the dataset:

#### ChEMBL
```bash
python download_chembl_dataset.py
```

or

```bash
python download_chembl_dataset.py -m number_of_molecules
```

#### ZINC

```bash
wget -i zinc_urls.txt -P ZINC
python process_zinc.py
python calculate_zinc_metrics.py
```

### Canocalize the SMILES strings:

```bash
python canonicalize_smiles.py
```

## Features

### Data Analysis
The dataset analysis script analyzes the downloaded dataset, providing insights into molecular properties and distributions some of which include:
- Molecular weight
- LogP (octanol-water partition coefficient)
- Number of hydrogen bond donors (HBD)
- Number of hydrogen bond acceptors (HBA)
- Number of atoms
- Number of heavy atoms
- Number of rings
- Topological Polar Surface Area (TPSA)
- Functional Group

```bash
python analyze_dataset.py
```

### Data Prepration