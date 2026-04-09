"""
HTGNN — Heterogeneous Temporal Graph Neural Network
Core model implementation using PyTorch Geometric.

Architecture:
    Input → Per-type Linear Projections → Temporal Self-Attention → MLP head
"""

import torch
import torch.nn as nn
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

        # 2. Temporal encoding
        self.time_enc = FourierTimeEncoding(hidden)
        self.temporal_proj = nn.Linear(hidden, hidden)

        # 3. Temporal self-attention
        self.temporal_attn = nn.MultiheadAttention(
            hidden, num_heads=4, dropout=dropout, batch_first=True
        )

        # Gating: combine graph embedding with temporal context
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )

        # 4. Fraud classification head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _aggregate_mean(
        self,
        source_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        num_dest_nodes: int,
    ) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return source_embeddings.new_zeros((num_dest_nodes, source_embeddings.size(-1)))

        source_index = edge_index[0].long()
        dest_index = edge_index[1].long()
        aggregated = source_embeddings.new_zeros((num_dest_nodes, source_embeddings.size(-1)))
        aggregated.index_add_(0, dest_index, source_embeddings[source_index])

        counts = source_embeddings.new_zeros(num_dest_nodes)
        counts.index_add_(0, dest_index, torch.ones_like(dest_index, dtype=source_embeddings.dtype))
        return aggregated / counts.clamp_min(1.0).unsqueeze(-1)

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
        delta_t: Optional[torch.Tensor] = None,    # [B, seq_len] time deltas
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        # ── 1. Type projection ──
        h = self.proj(x_dict)
        txn_emb = h["txn"]

        graph_contexts = []
        card_txn_edge = edge_index_dict.get(("card", "makes", "txn"))
        if card_txn_edge is not None and "card" in h:
            graph_contexts.append(self._aggregate_mean(h["card"], card_txn_edge, txn_emb.size(0)))

        txn_merchant_edge = edge_index_dict.get(("txn", "at", "merchant"))
        if txn_merchant_edge is not None and "merchant" in h:
            graph_contexts.append(
                self._aggregate_mean(h["merchant"], txn_merchant_edge.flip(0), txn_emb.size(0))
            )

        txn_device_edge = edge_index_dict.get(("txn", "via", "device"))
        if txn_device_edge is not None and "device" in h:
            graph_contexts.append(
                self._aggregate_mean(h["device"], txn_device_edge.flip(0), txn_emb.size(0))
            )

        if edge_index_dict.get(("card", "shared_device", "card")) is not None and "card" in h:
            shared_card_context = self._aggregate_mean(
                h["card"],
                edge_index_dict[("card", "shared_device", "card")],
                h["card"].size(0),
            )
            if card_txn_edge is not None:
                graph_contexts.append(self._aggregate_mean(shared_card_context, card_txn_edge, txn_emb.size(0)))

        if graph_contexts:
            graph_ctx = torch.stack(graph_contexts, dim=0).mean(dim=0)
        else:
            graph_ctx = torch.zeros_like(txn_emb)

        # ── 2. Temporal attention ──
        temporal_ctx = torch.zeros_like(txn_emb)
        if txn_seq is not None and delta_t is not None:
            seq = self.temporal_proj(txn_seq) + self.time_enc(delta_t)
            seq, _ = self.temporal_attn(seq, seq, seq)
            temporal_ctx = seq.mean(dim=1)

        ctx = graph_ctx + temporal_ctx

        # Gate: how much temporal context to use
        g = self.gate(torch.cat([txn_emb, ctx], dim=-1))
        fused = torch.cat([txn_emb, g * ctx], dim=-1)    # [B, hidden*2]

        if return_embeddings:
            return fused

        # ── 4. Classification ──
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
