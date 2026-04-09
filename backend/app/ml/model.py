"""
HTGNN — Heterogeneous Temporal Graph Neural Network
Core model implementation using PyTorch Geometric.

Architecture:
  Input -> Per-type Linear Projections -> HANConv x 2 -> Temporal Attention -> MLP head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData
from typing import Optional, Dict, Tuple
import math


class FourierTimeEncoding(nn.Module):
    """Learnable Fourier feature time encoding for temporal context."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w = nn.Linear(1, d_model // 2)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [...] -> [..., d_model]
        t = delta_t.unsqueeze(-1).float()
        freqs = self.w(t)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class TypeProjection(nn.Module):
    """Per-type linear projection with BN + activation."""

    def __init__(self, in_channels: Dict[str, int], hidden: int):
        super().__init__()
        self.layers = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(in_ch, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for ntype, in_ch in in_channels.items()
        })

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.layers[k](v) for k, v in x_dict.items() if k in self.layers}


class HTGNN(nn.Module):
    """
    Heterogeneous Temporal Graph Neural Network for fraud detection.

    Args:
        metadata:     PyG HeteroData metadata (node_types, edge_types)
        in_channels:  {node_type: feature_dim} dict
        hidden:       hidden dimension (default 128)
        heads:        attention heads in HANConv (default 4)
        dropout:      dropout rate (default 0.3)
        seq_len:      length of temporal transaction sequence (default 10)
    """

    def __init__(
        self,
        metadata: Tuple,
        in_channels: Dict[str, int],
        hidden: int = 128,
        heads: int = 4,
        dropout: float = 0.3,
        seq_len: int = 10,
    ):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len

        # 1. Type-specific projections
        self.proj = TypeProjection(in_channels, hidden)

        # 2. Heterogeneous attention convolutions
        self.conv1 = HANConv(hidden, hidden, metadata, heads=heads, dropout=dropout)
        self.conv2 = HANConv(hidden, hidden, metadata, heads=heads, dropout=dropout)

        # Layer norms after each conv
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # 3. Temporal encoding
        self.time_enc = FourierTimeEncoding(hidden)
        self.temporal_proj = nn.Linear(hidden, hidden)

        # 4. Temporal self-attention
        self.temporal_attn = nn.MultiheadAttention(
            hidden, num_heads=4, dropout=dropout, batch_first=True
        )

        # Gating: combine graph embedding with temporal context
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )

        # 5. Fraud classification head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        txn_seq: Optional[torch.Tensor] = None,   # [B, seq_len, hidden]
        delta_t: Optional[torch.Tensor] = None,   # [B, seq_len] time deltas
        seq_mask: Optional[torch.Tensor] = None,  # [B, seq_len] padding mask
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        # -- 1. Type projection --
        h = self.proj(x_dict)

        # -- 2. Graph message passing --
        h1 = self.conv1(h, edge_index_dict)
        h1 = {k: self.norm1(F.relu(v)) for k, v in h1.items()}

        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: self.norm2(v) for k, v in h2.items()}

        # Residual connection on txn node
        txn_emb = h2["txn"] + h1.get("txn", torch.zeros_like(h2["txn"]))

        # -- 3. Temporal attention --
        if txn_seq is not None and delta_t is not None:
            t_enc = self.time_enc(delta_t)                # [B, seq, hidden]
            seq = self.temporal_proj(txn_seq) + t_enc     # [B, seq, hidden]

            query = txn_emb.unsqueeze(1)                  # [B, 1, hidden]
            key_padding_mask = seq_mask.bool() if seq_mask is not None else None
            valid_hist = None
            if key_padding_mask is not None:
                # All-masked rows can make attention softmax produce NaNs.
                valid_hist = (~key_padding_mask).any(dim=1)
                if not bool(valid_hist.all()):
                    key_padding_mask = key_padding_mask.clone()
                    seq = seq.clone()
                    invalid_rows = ~valid_hist
                    key_padding_mask[invalid_rows, 0] = False
                    seq[invalid_rows, 0, :] = 0.0

            ctx, _ = self.temporal_attn(
                query,
                seq,
                seq,
                key_padding_mask=key_padding_mask,
            )
            ctx = ctx.squeeze(1)                          # [B, hidden]

            if valid_hist is not None and not bool(valid_hist.all()):
                ctx = ctx.clone()
                ctx[~valid_hist] = 0.0
        else:
            ctx = torch.zeros_like(txn_emb)

        # Gate: how much temporal context to use
        g = self.gate(torch.cat([txn_emb, ctx], dim=-1))
        fused = torch.cat([txn_emb, g * ctx], dim=-1)    # [B, hidden*2]

        if return_embeddings:
            return fused

        # -- 4. Classification --
        logits = self.head(fused).squeeze(-1)
        return logits

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        logits = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)


def build_model(metadata, in_channels: Dict[str, int], **kwargs) -> HTGNN:
    """Factory function to build HTGNN with sensible defaults."""
    return HTGNN(
        metadata=metadata,
        in_channels=in_channels,
        hidden=kwargs.get("hidden", 128),
        heads=kwargs.get("heads", 4),
        dropout=kwargs.get("dropout", 0.3),
        seq_len=kwargs.get("seq_len", 10),
    )
