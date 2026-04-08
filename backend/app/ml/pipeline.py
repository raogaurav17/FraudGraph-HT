"""
Multi-dataset data pipeline for FraudGraph.

Loads and preprocesses:
  - IEEE-CIS (primary: heterogeneous, temporal)
  - PaySim (synthetic: account graph, large scale)
  - Elliptic Bitcoin (temporal GNN benchmark)
  - YelpChi / FraudAmazon (heterogeneous GNN benchmarks via DGL)

Outputs unified PyG HeteroData objects ready for HTGNN training.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    train: HeteroData
    val: HeteroData
    test: HeteroData
    metadata: dict


# ─────────────────────────────────────────────────────────────────
# IEEE-CIS PIPELINE
# ─────────────────────────────────────────────────────────────────

class IEEECISPipeline:
    """
    Builds HeteroData from IEEE-CIS Fraud Detection dataset.
    Node types: card, merchant, device, txn
    Edge types: card→txn, txn→merchant, txn→device, card→card (shared device)
    """

    CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"]
    TXN_NUM_COLS = ["TransactionAmt", "dist1", "dist2"]
    TXN_CAT_COLS = ["ProductCD", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
    V_COLS = [f"V{i}" for i in range(1, 100)]  # first 99 V-features

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_raw(self) -> pd.DataFrame:
        tx_path = self.data_dir / "train_transaction.csv"
        id_path = self.data_dir / "train_identity.csv"

        if not tx_path.exists():
            raise FileNotFoundError(f"IEEE-CIS data not found at {tx_path}. See data/README.md")

        log.info("Loading IEEE-CIS transaction data...")
        tx = pd.read_csv(tx_path)
        id_df = pd.read_csv(id_path)
        df = tx.merge(id_df, on="TransactionID", how="left")
        log.info(f"  Loaded {len(df):,} transactions, fraud rate: {df['isFraud'].mean():.3%}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Card proxy identity
        df["card_id"] = df[self.CARD_COLS].fillna("NA").astype(str).agg("-".join, axis=1)

        # Log-scale amount
        df["log_amount"] = np.log1p(df["TransactionAmt"])

        # Hour of day (circular)
        # TransactionDT is seconds elapsed from reference
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Rolling card-level aggregates (simplified — use last known)
        df = df.sort_values("TransactionDT")
        df["card_txn_seq"] = df.groupby("card_id").cumcount()

        # Fill missing V columns
        for col in self.V_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Fill categorical
        for col in self.TXN_CAT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN")

        return df

    def build_hetero_data(self, df: pd.DataFrame) -> HeteroData:
        data = HeteroData()

        # ── Node indices ──
        card_ids, card_map = pd.factorize(df["card_id"])
        merch_ids, merch_map = pd.factorize(df["ProductCD"].fillna("OTHER"))
        device_ids, device_map = pd.factorize(df["DeviceInfo"].fillna("unknown"))
        txn_ids = np.arange(len(df))

        # ── Transaction features ──
        txn_feat_cols = ["log_amount", "hour_sin", "hour_cos"] + [
            c for c in self.V_COLS if c in df.columns
        ]
        txn_feats = df[txn_feat_cols].fillna(0).values.astype(np.float32)
        data["txn"].x = torch.tensor(txn_feats)

        # ── Card features (aggregate per card) ──
        card_feat = df.groupby("card_id").agg(
            avg_amount=("TransactionAmt", "mean"),
            txn_count=("TransactionID", "count"),
            std_amount=("TransactionAmt", "std"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(card_map).fillna(0).values.astype(np.float32)
        data["card"].x = torch.tensor(card_feat)

        # ── Merchant features ──
        merch_feat = df.groupby("ProductCD").agg(
            avg_amount=("TransactionAmt", "mean"),
            txn_count=("TransactionID", "count"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(merch_map).fillna(0).values.astype(np.float32)
        data["merchant"].x = torch.tensor(merch_feat)

        # ── Device features ──
        device_feat = df.groupby("DeviceInfo").agg(
            txn_count=("TransactionID", "count"),
            unique_cards=("card_id", "nunique"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(device_map).fillna(0).values.astype(np.float32)
        data["device"].x = torch.tensor(device_feat)

        # ── Edge indices ──
        # card → txn
        data["card", "makes", "txn"].edge_index = torch.tensor(
            [card_ids, txn_ids], dtype=torch.long
        )
        data["card", "makes", "txn"].time = torch.tensor(
            df["TransactionDT"].values, dtype=torch.long
        )

        # txn → merchant
        data["txn", "at", "merchant"].edge_index = torch.tensor(
            [txn_ids, merch_ids], dtype=torch.long
        )

        # txn → device (only where device exists)
        valid_dev = device_ids >= 0
        data["txn", "via", "device"].edge_index = torch.tensor(
            [txn_ids[valid_dev], device_ids[valid_dev]], dtype=torch.long
        )

        # card → card (shared device — fraud ring signal)
        card_device_df = df[["card_id", "DeviceInfo", "TransactionDT"]].dropna()
        shared = card_device_df.merge(card_device_df, on="DeviceInfo", suffixes=("_a", "_b"))
        shared = shared[shared["card_id_a"] != shared["card_id_b"]]
        shared = shared[abs(shared["TransactionDT_a"] - shared["TransactionDT_b"]) < 86400]

        if len(shared) > 0:
            src_cards = pd.factorize(shared["card_id_a"], sort=True)[0]
            dst_cards = pd.factorize(shared["card_id_b"], sort=True)[0]
            data["card", "shared_device", "card"].edge_index = torch.tensor(
                [src_cards, dst_cards], dtype=torch.long
            )

        # ── Labels ──
        data["txn"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)
        data["txn"].time = torch.tensor(df["TransactionDT"].values, dtype=torch.long)

        return data

    def temporal_split(
        self, data: HeteroData, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = data["txn"].time.numpy()
        train_end = np.quantile(t, train_ratio)
        val_end = np.quantile(t, train_ratio + val_ratio)
        train_mask = torch.tensor(t < train_end, dtype=torch.bool)
        val_mask = torch.tensor((t >= train_end) & (t < val_end), dtype=torch.bool)
        test_mask = torch.tensor(t >= val_end, dtype=torch.bool)
        return train_mask, val_mask, test_mask

    def run(self) -> DatasetSplit:
        df = self.load_raw()
        df = self.engineer_features(df)
        data = self.build_hetero_data(df)
        train_mask, val_mask, test_mask = self.temporal_split(data)
        data["txn"].train_mask = train_mask
        data["txn"].val_mask = val_mask
        data["txn"].test_mask = test_mask

        log.info(f"  Graph: {data}")
        log.info(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

        return DatasetSplit(
            train=data, val=data, test=data,
            metadata={
                "source": "ieee_cis",
                "n_txns": len(df),
                "fraud_rate": float(df["isFraud"].mean()),
                "n_card_nodes": data["card"].x.shape[0],
                "n_merchant_nodes": data["merchant"].x.shape[0],
            }
        )


# ─────────────────────────────────────────────────────────────────
# PAYSIM PIPELINE
# ─────────────────────────────────────────────────────────────────

class PaySimPipeline:
    """
    Builds account-level graph from PaySim synthetic mobile money dataset.
    Fraud only exists in CASH_OUT and TRANSFER transactions.
    Nodes: account (sender/receiver), txn
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_raw(self) -> pd.DataFrame:
        path = self.data_dir / "PS_20174392719_1491204439457_log.csv"
        if not path.exists():
            raise FileNotFoundError(f"PaySim data not found at {path}")

        log.info("Loading PaySim data...")
        df = pd.read_csv(path)
        # Filter to fraud-relevant transaction types
        df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
        log.info(f"  Loaded {len(df):,} relevant transactions, fraud rate: {df['isFraud'].mean():.4%}")
        return df

    def build_hetero_data(self, df: pd.DataFrame) -> HeteroData:
        data = HeteroData()

        df = df.copy()
        df["log_amount"] = np.log1p(df["amount"])
        df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
        df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

        # Node indices
        all_accounts = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
        acct_map = {name: i for i, name in enumerate(all_accounts)}
        n_accounts = len(all_accounts)
        txn_ids = np.arange(len(df))

        # Account features (aggregate)
        orig_feats = df.groupby("nameOrig").agg(
            sent_count=("step", "count"),
            avg_sent=("amount", "mean"),
            total_sent=("amount", "sum"),
        ).reindex(all_accounts).fillna(0).values.astype(np.float32)
        data["account"].x = torch.tensor(orig_feats)

        # Transaction features
        txn_feats = df[["log_amount", "balance_diff_orig", "balance_diff_dest", "balance_ratio"]].fillna(0)
        type_dummies = pd.get_dummies(df["type"], prefix="type").astype(np.float32)
        txn_feats = pd.concat([txn_feats, type_dummies], axis=1).values.astype(np.float32)
        data["txn"].x = torch.tensor(txn_feats)

        # Edges
        src = torch.tensor([acct_map[n] for n in df["nameOrig"]], dtype=torch.long)
        dst = torch.tensor([acct_map[n] for n in df["nameDest"]], dtype=torch.long)
        data["account", "sends", "txn"].edge_index = torch.tensor([src, txn_ids], dtype=torch.long)
        data["txn", "receives", "account"].edge_index = torch.tensor([txn_ids, dst], dtype=torch.long)

        data["txn"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)
        data["txn"].time = torch.tensor(df["step"].values, dtype=torch.long)

        return data

    def run(self) -> DatasetSplit:
        df = self.load_raw()
        data = self.build_hetero_data(df)
        t = data["txn"].time.numpy()
        train_mask = torch.tensor(t < np.quantile(t, 0.7), dtype=torch.bool)
        val_mask = torch.tensor((t >= np.quantile(t, 0.7)) & (t < np.quantile(t, 0.85)), dtype=torch.bool)
        test_mask = torch.tensor(t >= np.quantile(t, 0.85), dtype=torch.bool)
        data["txn"].train_mask = train_mask
        data["txn"].val_mask = val_mask
        data["txn"].test_mask = test_mask
        return DatasetSplit(train=data, val=data, test=data, metadata={"source": "paysim"})


# ─────────────────────────────────────────────────────────────────
# ELLIPTIC PIPELINE
# ─────────────────────────────────────────────────────────────────

class EllipticPipeline:
    """
    Loads Elliptic Bitcoin dataset.
    Homogeneous graph: 49 temporal time steps.
    Nodes are Bitcoin transactions, edges are BTC flows.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def run(self) -> DatasetSplit:
        feats_path = self.data_dir / "elliptic_txs_features.csv"
        edges_path = self.data_dir / "elliptic_txs_edgelist.csv"
        classes_path = self.data_dir / "elliptic_txs_classes.csv"

        if not feats_path.exists():
            raise FileNotFoundError(f"Elliptic data not found at {feats_path}")

        log.info("Loading Elliptic Bitcoin dataset...")
        feats = pd.read_csv(feats_path, header=None)
        edges = pd.read_csv(edges_path)
        classes = pd.read_csv(classes_path)

        # Column 0 = txId, 1 = time_step, 2: = features
        feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
        classes["label_bin"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1})

        merged = feats.merge(classes, on="txId", how="left")
        labeled = merged[merged["label_bin"] >= 0].copy()

        log.info(f"  Labeled: {len(labeled):,}, illicit rate: {labeled['label_bin'].mean():.3%}")

        # Build as hetero with single node type for compatibility
        data = HeteroData()
        x = torch.tensor(labeled.filter(like="f").values.astype(np.float32))
        data["txn"].x = x
        data["txn"].y = torch.tensor(labeled["label_bin"].values, dtype=torch.long)
        data["txn"].time = torch.tensor(labeled["time_step"].values, dtype=torch.long)

        # Add edges (within labeled nodes only)
        txid_to_idx = {txid: i for i, txid in enumerate(labeled["txId"])}
        valid_edges = edges[
            edges["txId1"].isin(txid_to_idx) & edges["txId2"].isin(txid_to_idx)
        ]
        src = torch.tensor([txid_to_idx[i] for i in valid_edges["txId1"]], dtype=torch.long)
        dst = torch.tensor([txid_to_idx[i] for i in valid_edges["txId2"]], dtype=torch.long)
        data["txn", "flows_to", "txn"].edge_index = torch.tensor([src.numpy(), dst.numpy()], dtype=torch.long)

        t = labeled["time_step"].values
        data["txn"].train_mask = torch.tensor(t <= 34, dtype=torch.bool)
        data["txn"].val_mask = torch.tensor((t > 34) & (t <= 42), dtype=torch.bool)
        data["txn"].test_mask = torch.tensor(t > 42, dtype=torch.bool)

        return DatasetSplit(
            train=data, val=data, test=data,
            metadata={"source": "elliptic", "n_labeled": len(labeled)}
        )


# ─────────────────────────────────────────────────────────────────
# YELP-CHI / FRAUD AMAZON (via DGL)
# ─────────────────────────────────────────────────────────────────

def load_dgl_fraud_dataset(name: str = "yelp") -> Optional[DatasetSplit]:
    """Load YelpChi or FraudAmazon from DGL."""
    try:
        import dgl
        from dgl.data import FraudDataset

        log.info(f"Loading {name} dataset from DGL...")
        dataset = FraudDataset(name)
        graph = dataset[0]

        feat = torch.tensor(graph.ndata["feature"].numpy().astype(np.float32))
        label = torch.tensor(graph.ndata["label"].numpy(), dtype=torch.long)

        data = HeteroData()
        data["review"].x = feat
        data["review"].y = label

        # Add edges for each relation
        rel_map = {
            "yelp": ["net_rsr", "net_rtr", "net_rur"],
            "amazon": ["net_upu", "net_usu", "net_uvu"],
        }
        node_type = "review" if name == "yelp" else "user"
        data["review"].x = feat

        for rel in rel_map.get(name, []):
            if graph.has_edges_of_type(rel):
                src, dst = graph.edges(etype=rel)
                data[node_type, rel, node_type].edge_index = torch.stack([src, dst], dim=0)

        train_mask = graph.ndata.get("train_mask", torch.zeros(len(label), dtype=torch.bool))
        val_mask = graph.ndata.get("val_mask", torch.zeros(len(label), dtype=torch.bool))
        test_mask = graph.ndata.get("test_mask", torch.ones(len(label), dtype=torch.bool))
        data[node_type].train_mask = train_mask
        data[node_type].val_mask = val_mask
        data[node_type].test_mask = test_mask

        fraud_rate = float(label.float().mean())
        log.info(f"  {name}: {len(label):,} nodes, fraud rate: {fraud_rate:.3%}")

        return DatasetSplit(
            train=data, val=data, test=data,
            metadata={"source": name, "fraud_rate": fraud_rate}
        )
    except ImportError:
        log.warning("DGL not installed — skipping YelpChi/FraudAmazon")
        return None
    except Exception as e:
        log.error(f"Failed to load {name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# UNIFIED PIPELINE
# ─────────────────────────────────────────────────────────────────

def load_all_datasets(data_dir: str, datasets: List[str] = None) -> Dict[str, DatasetSplit]:
    """Load and return all available datasets."""
    if datasets is None:
        datasets = ["ieee_cis", "paysim", "elliptic", "yelp", "amazon"]

    results = {}

    pipeline_map = {
        "ieee_cis": lambda: IEEECISPipeline(f"{data_dir}/ieee_cis").run(),
        "paysim":   lambda: PaySimPipeline(f"{data_dir}/paysim").run(),
        "elliptic": lambda: EllipticPipeline(f"{data_dir}/elliptic").run(),
        "yelp":     lambda: load_dgl_fraud_dataset("yelp"),
        "amazon":   lambda: load_dgl_fraud_dataset("amazon"),
    }

    for name in datasets:
        if name not in pipeline_map:
            log.warning(f"Unknown dataset: {name}")
            continue
        try:
            result = pipeline_map[name]()
            if result is not None:
                results[name] = result
                log.info(f"✓ Loaded dataset: {name}")
        except FileNotFoundError as e:
            log.warning(f"Skipping {name}: {e}")
        except Exception as e:
            log.error(f"Failed to load {name}: {e}")

    return results
