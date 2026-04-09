"""
HTGNN — heterogeneous temporal graph model optimized for lower memory footprint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class FourierTimeEncoding(nn.Module):
    """Learnable Fourier feature time encoding for temporal context."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w = nn.Linear(1, d_model // 2)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [...] → [..., d_model]
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
                nn.LayerNorm(hidden),
                nn.GELU(),
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
        self.metadata = metadata

        txn_in_dim = int(in_channels.get("txn", hidden))
        self.proj = TypeProjection(in_channels, hidden)

        self.use_han = False
        self.han1 = None
        self.han2 = None
        try:
            from torch_geometric.nn import HANConv

            self.han1 = HANConv(hidden, hidden, metadata, heads=max(1, heads), dropout=dropout)
            self.han2 = HANConv(hidden, hidden, metadata, heads=max(1, heads), dropout=dropout)
            self.use_han = True
        except Exception:
            self.use_han = False

        self.graph_norm1 = nn.LayerNorm(hidden)
        self.graph_norm2 = nn.LayerNorm(hidden)

        self.seq_input = nn.Linear(txn_in_dim, hidden)
        self.time_enc = FourierTimeEncoding(hidden)
        self.temporal_proj = nn.Linear(hidden, hidden)
        self.temporal_attn = nn.MultiheadAttention(
            hidden,
            num_heads=max(1, heads),
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _encode_graph(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, torch.Tensor]:
        proj = self.proj(x_dict)

        if not self.use_han or self.han1 is None or self.han2 is None:
            return proj

        edge_subset = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type in proj and dst_type in proj:
                edge_subset[edge_type] = edge_index

        if not edge_subset:
            return proj

        try:
            h = self.han1(proj, edge_subset)
            h = {
                k: F.relu(self.graph_norm1(v))
                for k, v in h.items()
                if v is not None
            }
            if not h:
                return proj

            h = self.han2(h, edge_subset)
            h = {
                k: self.graph_norm2(v)
                for k, v in h.items()
                if v is not None
            }
            return h or proj
        except Exception:
            # If heterogeneous attention fails for sparse/incomplete toy inputs,
            # keep inference alive by using projected node features.
            return proj

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
        txn_seq: Optional[torch.Tensor] = None,   # [N, seq_len, input_dim]
        delta_t: Optional[torch.Tensor] = None,   # [N, seq_len] time deltas
        seq_mask: Optional[torch.Tensor] = None,  # [N, seq_len] true for padding
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        txn_x = x_dict.get("txn")
        if txn_x is None:
            raise ValueError("x_dict must contain 'txn' node features")

        graph_dict = self._encode_graph(x_dict, edge_index_dict)
        h1_txn = graph_dict.get("txn")
        if h1_txn is None:
            # No graph context available, fall back to projected txn input.
            h1_txn = self.proj({"txn": txn_x}).get("txn")

        txn_emb = h1_txn

        if txn_seq is not None and delta_t is not None and txn_seq.ndim == 3:
            seq = self.seq_input(txn_seq.float()) + self.time_enc(delta_t.float())
            seq = self.temporal_proj(seq)

            key_padding_mask = seq_mask.bool() if seq_mask is not None else None
            valid_hist = None
            if key_padding_mask is not None:
                # If a row has all positions masked, attention softmax can produce NaN.
                valid_hist = (~key_padding_mask).any(dim=1)
                if not bool(valid_hist.all()):
                    key_padding_mask = key_padding_mask.clone()
                    seq = seq.clone()
                    invalid_rows = ~valid_hist
                    key_padding_mask[invalid_rows, 0] = False
                    seq[invalid_rows, 0, :] = 0.0

            query = txn_emb.unsqueeze(1)
            ctx, _ = self.temporal_attn(query, seq, seq, key_padding_mask=key_padding_mask)
            ctx = ctx.squeeze(1)

            if valid_hist is not None and not bool(valid_hist.all()):
                ctx = ctx.clone()
                ctx[~valid_hist] = 0.0
        else:
            ctx = torch.zeros_like(txn_emb)

        gate = self.gate(torch.cat([txn_emb, ctx], dim=-1))
        fused = torch.cat([txn_emb, gate * ctx], dim=-1)

        if return_embeddings:
            return fused

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
