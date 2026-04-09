"""
Multi-dataset data pipeline for FraudGraph.

Loads and preprocesses:
  - IEEE-CIS (primary: heterogeneous, temporal)
  - PaySim (synthetic: account graph, large scale)
  - Elliptic Bitcoin (temporal GNN benchmark)

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
from collections import defaultdict
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
    TXN_CAT_COLS = [
        "ProductCD", "P_emaildomain", "R_emaildomain",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    ]
    V_COLS = [f"V{i}" for i in range(1, 100)]  # first 99 V-features

    def __init__(self, data_dir: str, max_rows: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows

    def load_raw(self) -> pd.DataFrame:
        tx_path = self.data_dir / "train_transaction.csv"
        id_path = self.data_dir / "train_identity.csv"

        if not tx_path.exists():
            raise FileNotFoundError(f"IEEE-CIS data not found at {tx_path}. See data/README.md")

        tx_usecols = [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "isFraud",
            "P_emaildomain",
            "R_emaildomain",
            *[col for col in self.TXN_NUM_COLS if col != "TransactionAmt"],
            *self.CARD_COLS,
            *self.TXN_CAT_COLS,
            *self.V_COLS,
        ]

        log.info(
            "Loading IEEE-CIS transaction data%s...",
            f" (max_rows={self.max_rows:,})" if self.max_rows else "",
        )
        tx = pd.read_csv(tx_path, usecols=lambda c: c in tx_usecols, nrows=self.max_rows)
        id_df = pd.read_csv(id_path, usecols=["TransactionID", "DeviceInfo"], nrows=self.max_rows)
        df = tx.merge(id_df, on="TransactionID", how="left")
        log.info(f"  Loaded {len(df):,} transactions, fraud rate: {df['isFraud'].mean():.3%}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Card proxy identity
        df["card_id"] = df[self.CARD_COLS].fillna("NA").astype(str).agg("-".join, axis=1)
        df["email_id"] = (
            df["P_emaildomain"].fillna("NA").astype(str)
            + "|"
            + df["R_emaildomain"].fillna("NA").astype(str)
        )
        df["device_id"] = df["DeviceInfo"].fillna("unknown").astype(str)
        df["addr_id"] = df["addr1"].fillna("NA").astype(str)

        # Keep chronological order for rolling feature construction.
        df = df.sort_values(["card_id", "TransactionDT", "TransactionID"]).reset_index(drop=False)
        df = df.rename(columns={"index": "_orig_index"})

        n_rows = len(df)
        time_since_last = np.zeros(n_rows, dtype=np.float32)
        txn_count_1d = np.zeros(n_rows, dtype=np.float32)
        txn_count_7d = np.zeros(n_rows, dtype=np.float32)
        txn_count_30d = np.zeros(n_rows, dtype=np.float32)
        avg_amt_user = np.zeros(n_rows, dtype=np.float32)
        std_amt_user = np.zeros(n_rows, dtype=np.float32)
        unique_devices_per_user = np.zeros(n_rows, dtype=np.float32)
        unique_emails_per_card = np.zeros(n_rows, dtype=np.float32)
        card_device_count = np.zeros(n_rows, dtype=np.float32)
        card_addr_count = np.zeros(n_rows, dtype=np.float32)
        email_device_count = np.zeros(n_rows, dtype=np.float32)

        one_day = 86400.0
        seven_days = 7.0 * one_day
        thirty_days = 30.0 * one_day

        for _, group in df.groupby("card_id", sort=False):
            idx = group.index.to_numpy()
            times = group["TransactionDT"].to_numpy(dtype=np.float64, copy=False)
            amounts = group["TransactionAmt"].to_numpy(dtype=np.float32, copy=False)
            devices = group["device_id"].astype(str).to_numpy()
            emails = group["email_id"].astype(str).to_numpy()
            addrs = group["addr_id"].astype(str).to_numpy()

            n_group = len(group)
            if n_group == 0:
                continue

            if n_group > 1:
                time_since_last[idx[1:]] = np.diff(times).astype(np.float32, copy=False)

            count_before = np.arange(n_group, dtype=np.float32)
            prev_sum = np.cumsum(amounts, dtype=np.float64) - amounts
            prev_sq = np.cumsum(amounts ** 2, dtype=np.float64) - amounts ** 2
            avg_prev = np.zeros(n_group, dtype=np.float64)
            np.divide(prev_sum, count_before, out=avg_prev, where=count_before > 0)
            var_prev = np.zeros(n_group, dtype=np.float64)
            np.divide(
                prev_sq,
                count_before,
                out=var_prev,
                where=count_before > 1,
            )
            var_prev = var_prev - np.square(avg_prev)
            var_prev[count_before <= 1] = 0.0

            starts_1d = np.searchsorted(times, times - one_day, side="left")
            starts_7d = np.searchsorted(times, times - seven_days, side="left")
            starts_30d = np.searchsorted(times, times - thirty_days, side="left")
            positions = np.arange(n_group, dtype=np.float32)

            txn_count_1d[idx] = positions - starts_1d.astype(np.float32)
            txn_count_7d[idx] = positions - starts_7d.astype(np.float32)
            txn_count_30d[idx] = positions - starts_30d.astype(np.float32)
            avg_amt_user[idx] = avg_prev.astype(np.float32, copy=False)
            std_amt_user[idx] = np.sqrt(np.clip(var_prev, 0.0, None)).astype(np.float32, copy=False)

            seen_devices = set()
            seen_emails = set()
            seen_addrs = set()
            device_counts = defaultdict(int)
            addr_counts = defaultdict(int)
            email_device_counts = defaultdict(int)

            for pos, (device, email, addr) in enumerate(zip(devices, emails, addrs)):
                seen_devices.add(device)
                seen_emails.add(email)
                seen_addrs.add(addr)

                device_counts[device] += 1
                addr_counts[addr] += 1
                email_device_counts[(email, device)] += 1

                unique_devices_per_user[idx[pos]] = float(len(seen_devices))
                unique_emails_per_card[idx[pos]] = float(len(seen_emails))
                card_device_count[idx[pos]] = float(device_counts[device])
                card_addr_count[idx[pos]] = float(addr_counts[addr])
                email_device_count[idx[pos]] = float(email_device_counts[(email, device)])

        # Fill missing V columns
        for col in self.V_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        raw_numeric_cols = [
            "TransactionAmt",
            "dist1",
            "dist2",
            "card1",
            "card2",
            "card3",
            "card5",
            "addr1",
        ]
        for col in raw_numeric_cols:
            if col in df.columns and not isinstance(df[col], pd.DataFrame):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

        # Fill categorical
        for col in self.TXN_CAT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN")

        # Transaction-level engineered features for HTGNN input.
        feature_df = pd.DataFrame(
            {
                "log_amount": np.log1p(df["TransactionAmt"]),
                "hour": (df["TransactionDT"] // 3600) % 24,
                "hour_sin": np.sin(2 * np.pi * ((df["TransactionDT"] // 3600) % 24) / 24),
                "hour_cos": np.cos(2 * np.pi * ((df["TransactionDT"] // 3600) % 24) / 24),
                "time_since_last_txn": time_since_last,
                "txn_count_1d": txn_count_1d,
                "txn_count_7d": txn_count_7d,
                "txn_count_30d": txn_count_30d,
                "avg_amt_user": avg_amt_user,
                "std_amt_user": std_amt_user,
                "unique_devices_per_user": unique_devices_per_user,
                "unique_emails_per_card": unique_emails_per_card,
                "card_device_count": card_device_count,
                "card_addr_count": card_addr_count,
                "email_device_count": email_device_count,
                "card_txn_seq": df.groupby("card_id").cumcount().astype(np.float32),
            },
            index=df.index,
        )

        df = pd.concat([df, feature_df], axis=1)

        # Return original order for downstream graph construction.
        df = df.sort_values("_orig_index").drop(columns=["_orig_index"]).reset_index(drop=True)

        return df

    def build_hetero_data(self, df: pd.DataFrame) -> HeteroData:
        data = HeteroData()

        # ── Node indices ──
        card_ids, card_map = pd.factorize(df["card_id"])
        card_index_map = pd.Series(
            np.arange(len(card_map), dtype=np.int64),
            index=card_map,
        )
        merch_ids, merch_map = pd.factorize(df["ProductCD"].fillna("OTHER"))
        device_ids, device_map = pd.factorize(df["DeviceInfo"].fillna("unknown"))
        txn_ids = np.arange(len(df))

        # ── Transaction features ──
        txn_feat_cols = [
            "log_amount",
            "hour_sin",
            "hour_cos",
            "time_since_last_txn",
            "txn_count_1d",
            "txn_count_7d",
            "txn_count_30d",
            "avg_amt_user",
            "std_amt_user",
            "unique_devices_per_user",
            "unique_emails_per_card",
            "card_device_count",
            "card_addr_count",
            "email_device_count",
            "card_txn_seq",
        ] + [
            c for c in self.V_COLS if c in df.columns
        ]
        txn_feats = df[txn_feat_cols].fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["txn"].x = torch.from_numpy(txn_feats)

        # ── Card features (aggregate per card) ──
        card_feat = df.groupby("card_id").agg(
            avg_amount=("TransactionAmt", "mean"),
            txn_count=("TransactionID", "count"),
            std_amount=("TransactionAmt", "std"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(card_map).fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["card"].x = torch.from_numpy(card_feat)

        # ── Merchant features ──
        merch_feat = df.groupby("ProductCD").agg(
            avg_amount=("TransactionAmt", "mean"),
            txn_count=("TransactionID", "count"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(merch_map).fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["merchant"].x = torch.from_numpy(merch_feat)

        # ── Device features ──
        device_feat = df.groupby("DeviceInfo").agg(
            txn_count=("TransactionID", "count"),
            unique_cards=("card_id", "nunique"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(device_map).fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["device"].x = torch.from_numpy(device_feat)

        # ── Edge indices ──
        # card → txn
        card_txn_edges = np.stack([card_ids, txn_ids], axis=0).astype(np.int64, copy=False)
        data["card", "makes", "txn"].edge_index = torch.from_numpy(card_txn_edges)
        data["card", "makes", "txn"].time = torch.tensor(
            df["TransactionDT"].values, dtype=torch.long
        )

        # txn → merchant
        txn_merch_edges = np.stack([txn_ids, merch_ids], axis=0).astype(np.int64, copy=False)
        data["txn", "at", "merchant"].edge_index = torch.from_numpy(txn_merch_edges)

        # txn → device (only where device exists)
        valid_dev = device_ids >= 0
        txn_dev_edges = np.stack([txn_ids[valid_dev], device_ids[valid_dev]], axis=0).astype(np.int64, copy=False)
        data["txn", "via", "device"].edge_index = torch.from_numpy(txn_dev_edges)

        # card → card (shared device — fraud ring signal)
        # Use grouped pair generation to avoid quadratic memory blowups from self-merges.
        max_cards_per_device = 50
        src_parts: List[np.ndarray] = []
        dst_parts: List[np.ndarray] = []

        grouped_cards = (
            df[["DeviceInfo", "card_id"]]
            .dropna()
            .drop_duplicates()
            .groupby("DeviceInfo", sort=False)["card_id"]
            .agg(list)
        )

        skipped_hubs = 0
        for cards in grouped_cards:
            n_cards = len(cards)
            if n_cards < 2:
                continue
            if n_cards > max_cards_per_device:
                skipped_hubs += 1
                continue

            idx = card_index_map.reindex(cards).dropna().to_numpy(dtype=np.int64, copy=False)
            n_idx = len(idx)
            if n_idx < 2:
                continue

            src = np.repeat(idx, n_idx)
            dst = np.tile(idx, n_idx)
            keep = src != dst
            src_parts.append(src[keep])
            dst_parts.append(dst[keep])

        if src_parts:
            card_card_edges = np.stack(
                [np.concatenate(src_parts), np.concatenate(dst_parts)],
                axis=0,
            ).astype(np.int64, copy=False)
            data["card", "shared_device", "card"].edge_index = torch.from_numpy(card_card_edges)

        if skipped_hubs:
            log.info(
                "Skipped %d high-cardinality devices when building shared_device edges",
                skipped_hubs,
            )

        # ── Labels ──
        data["txn"].y = torch.from_numpy(df["isFraud"].to_numpy(dtype=np.int64, copy=False))
        data["txn"].time = torch.from_numpy(df["TransactionDT"].to_numpy(dtype=np.int64, copy=False))

        return data

    def temporal_split(
        self, data: HeteroData, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = data["txn"].time.numpy()
        train_end = np.quantile(t, train_ratio)
        val_end = np.quantile(t, train_ratio + val_ratio)
        train_mask = torch.from_numpy(t < train_end)
        val_mask = torch.from_numpy((t >= train_end) & (t < val_end))
        test_mask = torch.from_numpy(t >= val_end)
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
        ).reindex(all_accounts).fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["account"].x = torch.from_numpy(orig_feats)

        # Transaction features
        txn_feats = df[["log_amount", "balance_diff_orig", "balance_diff_dest", "balance_ratio"]].fillna(0)
        type_dummies = pd.get_dummies(df["type"], prefix="type").astype(np.float32)
        txn_feats = pd.concat([txn_feats, type_dummies], axis=1).to_numpy(dtype=np.float32, copy=False)
        data["txn"].x = torch.from_numpy(txn_feats)

        # Edges
        src = df["nameOrig"].map(acct_map).to_numpy(dtype=np.int64, copy=False)
        dst = df["nameDest"].map(acct_map).to_numpy(dtype=np.int64, copy=False)
        acct_txn_edges = np.stack([src, txn_ids], axis=0).astype(np.int64, copy=False)
        txn_acct_edges = np.stack([txn_ids, dst], axis=0).astype(np.int64, copy=False)
        data["account", "sends", "txn"].edge_index = torch.from_numpy(acct_txn_edges)
        data["txn", "receives", "account"].edge_index = torch.from_numpy(txn_acct_edges)

        data["txn"].y = torch.from_numpy(df["isFraud"].to_numpy(dtype=np.int64, copy=False))
        data["txn"].time = torch.from_numpy(df["step"].to_numpy(dtype=np.int64, copy=False))

        return data

    def run(self) -> DatasetSplit:
        df = self.load_raw()
        data = self.build_hetero_data(df)
        t = data["txn"].time.numpy()
        q70 = np.quantile(t, 0.7)
        q85 = np.quantile(t, 0.85)
        train_mask = torch.from_numpy(t < q70)
        val_mask = torch.from_numpy((t >= q70) & (t < q85))
        test_mask = torch.from_numpy(t >= q85)
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
        x = torch.from_numpy(labeled.filter(like="f").to_numpy(dtype=np.float32, copy=False))
        data["txn"].x = x
        data["txn"].y = torch.from_numpy(labeled["label_bin"].to_numpy(dtype=np.int64, copy=False))
        data["txn"].time = torch.from_numpy(labeled["time_step"].to_numpy(dtype=np.int64, copy=False))

        # Add edges (within labeled nodes only)
        txid_to_idx = {txid: i for i, txid in enumerate(labeled["txId"])}
        valid_edges = edges[
            edges["txId1"].isin(txid_to_idx) & edges["txId2"].isin(txid_to_idx)
        ]
        src = valid_edges["txId1"].map(txid_to_idx).to_numpy(dtype=np.int64, copy=False)
        dst = valid_edges["txId2"].map(txid_to_idx).to_numpy(dtype=np.int64, copy=False)
        txn_edges = np.stack([src, dst], axis=0).astype(np.int64, copy=False)
        data["txn", "flows_to", "txn"].edge_index = torch.from_numpy(txn_edges)

        t = labeled["time_step"].to_numpy(copy=False)
        data["txn"].train_mask = torch.from_numpy(t <= 34)
        data["txn"].val_mask = torch.from_numpy((t > 34) & (t <= 42))
        data["txn"].test_mask = torch.from_numpy(t > 42)

        return DatasetSplit(
            train=data, val=data, test=data,
            metadata={"source": "elliptic", "n_labeled": len(labeled)}
        )


# ─────────────────────────────────────────────────────────────────
# UNIFIED PIPELINE
# ─────────────────────────────────────────────────────────────────

def load_all_datasets(
    data_dir: str,
    datasets: List[str] = None,
    ieee_max_rows: Optional[int] = None,
) -> Dict[str, DatasetSplit]:
    """Load and return all available datasets."""
    if datasets is None:
        datasets = ["ieee_cis", "paysim", "elliptic"]

    results = {}

    pipeline_map = {
        "ieee_cis": lambda: IEEECISPipeline(
            f"{data_dir}/ieee_cis", max_rows=ieee_max_rows
        ).run(),
        "paysim":   lambda: PaySimPipeline(f"{data_dir}/paysim").run(),
        "elliptic": lambda: EllipticPipeline(f"{data_dir}/elliptic").run(),
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
