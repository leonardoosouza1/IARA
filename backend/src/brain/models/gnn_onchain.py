"""
I.A.R.A. v11 — Sparse Graph Neural Network for On-Chain Analysis
=================================================================
Lightweight GCN (Graph Convolutional Network) that processes whale wallet
transaction graphs WITHOUT requiring torch-geometric.

Uses native torch.sparse for maximum ROCm GFX 10.3.0 compatibility.

Architecture:
  Node Features [N, 3] + Sparse Adj [N, N]
  → GCN Layer 1 [N, 32]
  → GCN Layer 2 [N, 32]
  → Global Mean Pooling [1, 32]
  → Linear Projection [1, output_dim]

Reference: Kipf & Welling (2017), "Semi-Supervised Classification with GCNs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseGCNLayer(nn.Module):
    """
    Manual Sparse Graph Convolutional Layer.

    Implements: H' = σ(D^{-1/2} A D^{-1/2} H W)

    Uses torch.sparse.mm for efficient computation on both
    CUDA (NVIDIA) and ROCm (AMD) backends.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_sparse: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, in_features] dense node feature matrix
            adj_sparse: [N, N] sparse adjacency matrix (COO format)
            n_nodes: Number of nodes

        Returns:
            [N, out_features] updated node features
        """
        # Add self-loops: A_hat = A + I
        identity_indices = torch.arange(n_nodes, device=node_features.device)
        identity_indices = torch.stack([identity_indices, identity_indices])
        identity_values = torch.ones(n_nodes, device=node_features.device)

        # Combine adjacency with self-loops
        if adj_sparse._nnz() > 0:
            combined_indices = torch.cat([adj_sparse.indices(), identity_indices], dim=1)
            combined_values = torch.cat([adj_sparse.values(), identity_values])
        else:
            combined_indices = identity_indices
            combined_values = identity_values

        a_hat = torch.sparse_coo_tensor(
            combined_indices, combined_values,
            size=(n_nodes, n_nodes),
            device=node_features.device,
        ).coalesce()

        # Compute degree matrix D^{-1/2}
        degree = torch.sparse.sum(a_hat, dim=1).to_dense()
        degree_inv_sqrt = torch.zeros_like(degree)
        mask = degree > 0
        degree_inv_sqrt[mask] = degree[mask].pow(-0.5)

        # D^{-1/2} A D^{-1/2}: Scale rows and columns
        d_inv = degree_inv_sqrt.unsqueeze(1)

        # H * W
        support = torch.mm(node_features, self.weight)

        # D^{-1/2} * (A_hat * (D^{-1/2} * support))
        scaled_support = support * d_inv
        output = torch.sparse.mm(a_hat, scaled_support)
        output = output * d_inv

        if self.bias is not None:
            output = output + self.bias

        return output


class OnChainGNN(nn.Module):
    """
    Lightweight 2-layer GCN for on-chain whale wallet analysis.

    Processes the transaction graph and produces a single graph-level
    embedding that captures accumulation/distribution patterns.
    """

    def __init__(
        self,
        node_features: int = 3,
        hidden_dim: int = 32,
        output_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn1 = SparseGCNLayer(node_features, hidden_dim)
        self.gcn2 = SparseGCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        node_features: torch.Tensor,
        adj_sparse: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, 3] node feature matrix
            adj_sparse: [N, N] sparse adjacency matrix

        Returns:
            [1, output_dim] graph-level embedding
        """
        n_nodes = node_features.size(0)

        # Ensure sparse tensor is on same device
        if adj_sparse.device != node_features.device:
            adj_sparse = adj_sparse.to(node_features.device)

        # GCN Layer 1
        h = self.gcn1(node_features, adj_sparse, n_nodes)
        h = F.silu(h)
        h = self.dropout(h)

        # GCN Layer 2
        h = self.gcn2(h, adj_sparse, n_nodes)
        h = F.silu(h)

        # Global Mean Pooling: [N, hidden] → [1, hidden]
        graph_embedding = h.mean(dim=0, keepdim=True)

        # Project to output dim
        return self.projection(graph_embedding)
