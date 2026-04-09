"""
HTGNN V2 — sequence-first fraud model with graph structural context.
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
        hidden:       hidden dimension (default 256)
        heads:        attention heads in HANConv (default 4)
        dropout:      dropout rate (default 0.2)
        seq_len:      length of temporal transaction sequence (default 15)
    """

    def __init__(
        self,
        metadata: Tuple,
        in_channels: Dict[str, int],
        hidden: int = 256,
        heads: int = 4,
        dropout: float = 0.2,
        seq_len: int = 15,
    ):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.metadata = metadata

        txn_in_dim = int(in_channels.get("txn", hidden))
        card_in_dim = int(in_channels.get("card", txn_in_dim))
        seq_in_dim = txn_in_dim

        self.txn_encoder = nn.Sequential(
            nn.Linear(txn_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        self.seq_input = nn.Linear(seq_in_dim, hidden)
        self.seq_time_enc = FourierTimeEncoding(hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, heads),
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))

        self.node_proj = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden)
            for ntype, dim in in_channels.items()
        })

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

        self.cross_attn = nn.MultiheadAttention(
            hidden,
            num_heads=max(1, heads),
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _encode_sequence(
        self,
        txn_seq: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor],
        seq_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if txn_seq is None or delta_t is None:
            return torch.zeros((batch_size, self.hidden), dtype=torch.float32, device=device)

        if txn_seq.ndim != 3:
            return torch.zeros((batch_size, self.hidden), dtype=torch.float32, device=device)

        x = self.seq_input(txn_seq.float()) + self.seq_time_enc(delta_t.float())
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        if seq_mask is None:
            seq_mask = torch.zeros((x.size(0), x.size(1) - 1), dtype=torch.bool, device=x.device)
        cls_mask = torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, seq_mask.bool()], dim=1)
        out = self.seq_encoder(x, src_key_padding_mask=full_mask)
        return out[:, 0]

    def _encode_graph(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, torch.Tensor]:
        proj = {
            ntype: self.node_proj[ntype](x)
            for ntype, x in x_dict.items()
            if ntype in self.node_proj
        }

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

        txn_emb = self.txn_encoder(txn_x)
        n_txn = txn_emb.size(0)

        seq_ctx = self._encode_sequence(
            txn_seq=txn_seq,
            delta_t=delta_t,
            seq_mask=seq_mask,
            batch_size=n_txn,
            device=txn_emb.device,
        )

        graph_dict = self._encode_graph(x_dict, edge_index_dict)
        graph_ctx = graph_dict.get("txn")
        if graph_ctx is None:
            graph_ctx = torch.zeros_like(txn_emb)

        context = torch.stack([seq_ctx, graph_ctx], dim=1)
        fused_attn, _ = self.cross_attn(
            txn_emb.unsqueeze(1),
            context,
            context,
        )
        fused_attn = fused_attn.squeeze(1)
        fused = torch.cat([txn_emb, fused_attn], dim=-1)

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
        hidden=kwargs.get("hidden", 256),
        heads=kwargs.get("heads", 4),
        dropout=kwargs.get("dropout", 0.2),
        seq_len=kwargs.get("seq_len", 15),
    )
