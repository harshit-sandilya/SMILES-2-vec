# SMILES Graph Neural Network Project

This project implements a Graph Neural Network (GNN) pipeline for molecular property prediction and representation learning using SMILES strings. The workflow includes data download, preprocessing, model training, embedding generation, visualization, and benchmarking against standard datasets.

## Features
- **Data Download**: Fetch molecules and properties from ChEMBL using `download.py`.
- **Preprocessing**: Filter, tokenize, and convert SMILES to graph data structures.
- **Model**: GNN based on PyTorch Geometric with attention layers for atom and bond prediction.
- **Training**: Masked atom and bond prediction for self-supervised learning (`train.py`).
- **Embeddings**: Generate molecular embeddings for downstream tasks (`generate_embeddings.py`).
- **Visualization**: UMAP and property scatter plots for embedding analysis (`visualise_molecules.py`).
- **Property Prediction**: Predict molecular properties (e.g., LogP) using embeddings and random forest regression (`logP_test.py`).
- **Benchmarking**: Compare GNN embeddings to ECFP fingerprints on MoleculeNet datasets.

## Installation
1. **Clone the repository** and install dependencies:
   ```bash
   git clone <repo-url>
   cd SMILES
   pip install -r requirements.txt
   ```

2. **Download ChEMBL Data**:
   ```bash
   python download.py -m 10000
   ```
   or to download complete dataset
   ```bash
   python download.py
   ```

## Usage

### 1. Training
- Train the GNN model:
  ```bash
  python train.py --train_file chembl_data.csv
  ```
  Model checkpoints are saved in the `model/` directory.

### 2. Generate Embeddings
- Generate molecular embeddings using the trained model:
  ```bash
  python generate_embeddings.py --model-file model/model_final.pt --data-file chembl_data.csv
  ```
  Embeddings are saved to `embeddings/molecule_embeddings.pkl`.

### 3. Visualization
- Visualize embeddings colored by molecular properties:
  ```bash
  python visualise_molecules.py --embeddings embeddings/molecule_embeddings.pkl --data chembl_data.csv
  ```
  UMAP plots are saved in the `results/` directory.

### 4. Property Prediction
- Predict LogP or other properties from embeddings:
  ```bash
  python logP_test.py --embeddings embeddings/molecule_embeddings.pkl --data chembl_data.csv
  ```
  Results and scatter plots are saved in `results/`.

### 5. Benchmarking

## File Overview
- `download.py`: Download molecules from ChEMBL.
- `preprocess.ipynb`: Data preprocessing and exploratory analysis.
- `tokenizer.py`: SMILES to graph tokenization.
- `dataset.py`: PyTorch Dataset for masked molecule modeling.
- `model.py`: GNN model definition.
- `train.py`: Model training script.
- `generate_embeddings.py`: Embedding generation script.
- `visualise_molecules.py`: UMAP and property visualization.
- `logP_test.py`: Property prediction from embeddings.
- `utils.py`: Utility functions.
- `chembl_data.csv`: Downloaded molecule data.
- `molecule_embeddings.pkl`: Generated embeddings.

## Citation
If you use this codebase, please cite the original ChEMBL and MoleculeNet papers, and any relevant GNN or PyTorch Geometric references.

## License
This project is for research and educational purposes. See `LICENSE` for details.
