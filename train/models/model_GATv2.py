import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GraphMoleculeModelGATv2(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        ATOM_VOCAB_SIZE,
        BOND_VOCAB_SIZE,
        embedding_dim=256,  # final molecule embedding dimension
        num_mol_props=4,  # logP, MolWt, TPSA, RingCount
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim  # final molecule embedding dimension
        self.mol_props = num_mol_props

        # ---------------- Embeddings ----------------
        ATOM_FEAT_DIM = 64

        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, ATOM_FEAT_DIM)
        self.degree_emb = nn.Embedding(11, ATOM_FEAT_DIM)
        self.charge_emb = nn.Embedding(9, ATOM_FEAT_DIM)
        self.hcount_emb = nn.Embedding(9, ATOM_FEAT_DIM)
        self.aromatic_emb = nn.Embedding(2, ATOM_FEAT_DIM)
        self.ring_emb = nn.Embedding(2, ATOM_FEAT_DIM)
        self.hybrid_emb = nn.Embedding(6, ATOM_FEAT_DIM)
        self.atom_input_proj = nn.Linear(ATOM_FEAT_DIM * 7, hidden_dim)

        # ----------------------------------------------------------------
        # Bond embeddings
        # 5 separate embedding tables, one per bond feature column:
        #   0: bond_type (1–4 + mask)   → BOND_VOCAB_SIZE entries
        #   1: is_conjugated (0–1)
        #   2: is_in_ring    (0–1)
        #   3: ring_size     (0–8)      → 9 entries
        #   4: stereo        (0–3)
        # All five concatenated and projected to hidden_dim so GATv2Conv
        # receives the same edge_dim it always expected.
        # ----------------------------------------------------------------
        BOND_FEAT_DIM = 32

        self.bond_type_emb = nn.Embedding(BOND_VOCAB_SIZE, BOND_FEAT_DIM)
        self.conjugated_emb = nn.Embedding(2, BOND_FEAT_DIM)
        self.bond_ring_emb = nn.Embedding(2, BOND_FEAT_DIM)
        self.ring_size_emb = nn.Embedding(9, BOND_FEAT_DIM)
        self.stereo_emb = nn.Embedding(4, BOND_FEAT_DIM)
        self.bond_input_proj = nn.Linear(BOND_FEAT_DIM * 5, hidden_dim)

        # ---------------- GATv2 layers ----------------
        head_dim = hidden_dim // num_heads

        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=num_heads,
                edge_dim=hidden_dim,
                concat=True,
                dropout=0.1,
            )
            self.gnn_layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # ---------------- SMART POOLING ----------------
        # Triple pooling: sum + max + mean → MLP projection
        # This creates a 3*hidden_dim dimensional embedding
        # Then project back to hidden_dim for final output
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # ----------------------------------------------------------------
        # Property prediction head  ← NEW
        # Takes the final graph embedding [B, embedding_dim] and predicts
        # 4 z-score-normalised molecular scalars: logP, MolWt, TPSA, Rings.
        #
        # This is the direct gradient signal the pooling path was missing.
        # Without this head, pool_projection receives zero gradient after
        # contrastive loss removal and stays at random initialisation for
        # the entire training run — producing arbitrary embedding geometry
        # regardless of how well atom/bond reconstruction is performing.
        #
        # Property loss weight is controlled by PROPERTY_LOSS_WEIGHT in
        # config_GATv2.py (default 0.5). Properties are z-score normalised
        # in optimise_graph.py so MSE is on a consistent scale across all 4.
        # ----------------------------------------------------------------
        self.property_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, num_mol_props),
        )

        # ---------------- Prediction heads ----------------
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)

        self.predict_bond = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, BOND_VOCAB_SIZE),
        )

    # ====================================================
    # Shared message-passing core
    # ====================================================
    def _message_passing(self, x, edge_index, edge_bond_feats):
        """
        Embed raw integer features, run GATv2 layers with residual
        connections and LayerNorm, return (node_embeddings, edge_embeddings).

        Parameters
        ----------
        x               : [N, 7]  long — atom feature matrix
        edge_index      : [2, E]  long — directed edge list
        edge_bond_feats : [E, 5]  long — bond feature matrix
        """
        x = x.long()
        atoms = self.atom_embedder(x[:, 0])
        degrees = self.degree_emb(x[:, 1])
        charges = self.charge_emb(x[:, 2])
        hcounts = self.hcount_emb(x[:, 3])
        aromatics = self.aromatic_emb(x[:, 4])
        rings = self.ring_emb(x[:, 5])
        hybridizations = self.hybrid_emb(x[:, 6])
        x = torch.cat(
            [atoms, degrees, charges, hcounts, aromatics, rings, hybridizations],
            dim=-1,
        )  # [N, ATOM_FEAT_DIM * 7]
        x = self.atom_input_proj(x)  # [N, hidden_dim]

        # Bond embedding — five lookups, one per feature column
        bf = edge_bond_feats.long()  # [E, 5]
        e = torch.cat(
            [
                self.bond_type_emb(bf[:, 0]),
                self.conjugated_emb(bf[:, 1]),
                self.bond_ring_emb(bf[:, 2]),
                self.ring_size_emb(bf[:, 3]),
                self.stereo_emb(bf[:, 4]),
            ],
            dim=-1,
        )  # [E, BOND_FEAT_DIM * 5]
        e = self.bond_input_proj(e)  # [E, hidden_dim]

        # GATv2 layers with residual + LayerNorm
        for conv, ln in zip(self.gnn_layers, self.layer_norms):
            x_normed = ln(x)
            x_update = F.elu(conv(x_normed, edge_index, e))
            x = x + x_update

        return x, e

    # ====================================================
    # Internal pooling helper
    # ====================================================
    def _pool(self, x, batch_idx, e, edge_index):
        """
        Combined atom + bond triple-pooling → MLP → L2 normalise.

        Atom triple-pool  [B, hidden*3]  — node-level chemistry
        Bond triple-pool  [B, hidden*3]  — edge-level chemistry
        Concatenated      [B, hidden*6]
        MLP               [B, hidden] → [B, embedding_dim]
        L2 normalise      → unit hypersphere

        Note: edge_batch assigns each edge to its source node's graph.
        Bidirectional storage means add_pool double-counts every bond —
        the MLP compensates. mean_pool and max_pool are unaffected.
        """
        atom_pool = torch.cat(
            [
                global_add_pool(x, batch_idx),
                global_max_pool(x, batch_idx),
                global_mean_pool(x, batch_idx),
            ],
            dim=-1,
        )  # [B, hidden_dim * 3]

        edge_batch = batch_idx[edge_index[0]]
        bond_pool = torch.cat(
            [
                global_add_pool(e, edge_batch),
                global_max_pool(e, edge_batch),
                global_mean_pool(e, edge_batch),
            ],
            dim=-1,
        )  # [B, hidden_dim * 3]

        combined = torch.cat([atom_pool, bond_pool], dim=-1)  # [B, hidden_dim * 6]
        molecule_embedding = self.pool_projection(combined)  # [B, embedding_dim]
        molecule_embedding = F.normalize(molecule_embedding, p=2, dim=-1)
        return molecule_embedding

    # ====================================================
    # Forward pass (used during masked training)
    # ====================================================
    def forward(self, data):
        x, e = self._message_passing(
            data.x,
            data.edge_index,
            data.edge_bond_feats,
        )

        # --- Atom prediction (unchanged) ---
        predicted_atom_logits = self.predict_atom(x)  # [N, ATOM_VOCAB_SIZE]

        # --- Bond prediction (CHANGE 4: includes edge embedding) ---
        row, col = data.edge_index
        atom_pair_features = torch.cat(
            [x[row], x[col], e], dim=-1
        )  # [E, hidden_dim * 3]
        predicted_bond_logits = self.predict_bond(
            atom_pair_features
        )  # [E, BOND_VOCAB_SIZE]

        # --- Graph embedding ---
        graph_emb = self._pool(x, data.batch, e, data.edge_index)  # [B, embedding_dim]

        predicted_props = self.property_head(graph_emb)  # [B, num_mol_props]

        return predicted_atom_logits, predicted_bond_logits, graph_emb, predicted_props

    # ====================================================
    # SMART POOLING - Multiple strategies combined
    # ====================================================
    def get_embedding(self, data):
        """
        calls _pool() which is now the same trained path as forward().
        Previously this called pool_projection on an untrained basis.
        """
        x, e = self._message_passing(
            data.x,
            data.edge_index,
            data.edge_bond_feats,
        )
        return self._pool(x, data.batch, e, data.edge_index)

    # ====================================================
    # Graph-level embedding (unified API)
    # ====================================================
    def get_graph_embedding(self, data):
        return self.get_embedding(data)