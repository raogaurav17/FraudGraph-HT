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
    V_COLS = [f"V{i}" for i in range(1, 340)]
    FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com"}

    def __init__(
        self,
        data_dir: str,
        max_rows: Optional[int] = None,
        max_v_features: int = 160,
        max_cards_per_device: int = 20,
        feature_mode: str = "enhanced",
    ):
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows
        self.max_v_features = max(0, min(max_v_features, len(self.V_COLS)))
        self.max_cards_per_device = max(2, max_cards_per_device)
        self.feature_mode = feature_mode
        self.seq_len = 10 if feature_mode == "original" else 15

    def load_raw(self) -> pd.DataFrame:
        tx_path = self.data_dir / "train_transaction.csv"
        id_path = self.data_dir / "train_identity.csv"

        if not tx_path.exists():
            raise FileNotFoundError(f"IEEE-CIS data not found at {tx_path}. See data/README.md")

        selected_v_cols = self.V_COLS[: self.max_v_features] if self.max_v_features else []
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
            *selected_v_cols,
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

        if self.feature_mode == "original":
            t = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            q70 = np.quantile(t, 0.7)
            q85 = np.quantile(t, 0.85)
            df["split"] = np.where(t < q70, "train", np.where(t < q85, "val", "test"))

            df["card_id"] = df[self.CARD_COLS].fillna("NA").astype(str).agg("-".join, axis=1)

            for col in ["ProductCD", "DeviceInfo", "P_emaildomain", "R_emaildomain"]:
                if col in df.columns:
                    df[col] = df[col].fillna("UNKNOWN").astype(str)

            for col in ["TransactionAmt", "dist1", "dist2"]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    median_val = s[df["split"] == "train"].median()
                    if pd.isna(median_val):
                        median_val = s.median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    df[col] = s.fillna(float(median_val))

            for col in self.V_COLS[: self.max_v_features]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    median_val = s[df["split"] == "train"].median()
                    if pd.isna(median_val):
                        median_val = s.median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    df[col] = s.fillna(float(median_val))

            hour = ((pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0) // 3600) % 24).astype(np.float32)

            # Restore a small set of high-signal behavior features for original mode.
            work = pd.DataFrame(
                {
                    "row_id": np.arange(len(df), dtype=np.int64),
                    "card_id": df["card_id"].astype(str),
                    "TransactionDT": pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0),
                    "TransactionID": pd.to_numeric(df["TransactionID"], errors="coerce").fillna(0),
                    "TransactionAmt": pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0),
                    "device_id": df["DeviceInfo"].fillna("UNKNOWN").astype(str),
                }
            ).sort_values(["card_id", "TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)

            n_rows = len(work)
            one_day = 86400.0
            seven_days = 7.0 * one_day
            txn_count_1d = np.zeros(n_rows, dtype=np.float32)
            amt_zscore_7d = np.zeros(n_rows, dtype=np.float32)
            is_new_device = np.zeros(n_rows, dtype=np.float32)

            for _, group in work.groupby("card_id", sort=False):
                idx = group.index.to_numpy()
                times = group["TransactionDT"].to_numpy(dtype=np.float64, copy=False)
                amts = group["TransactionAmt"].to_numpy(dtype=np.float64, copy=False)
                devs = group["device_id"].to_numpy(copy=False)
                n_group = len(group)
                if n_group == 0:
                    continue

                starts_1d = np.searchsorted(times, times - one_day, side="left")
                starts_7d = np.searchsorted(times, times - seven_days, side="left")
                positions = np.arange(n_group, dtype=np.float32)
                txn_count_1d[idx] = positions - starts_1d.astype(np.float32)

                right = np.arange(1, n_group + 1)
                left_7 = starts_7d.astype(np.int64)
                count_7 = right - left_7
                prefix_sum = np.concatenate([[0.0], np.cumsum(amts, dtype=np.float64)])
                prefix_sq = np.concatenate([[0.0], np.cumsum(amts * amts, dtype=np.float64)])
                sum_7 = prefix_sum[right] - prefix_sum[left_7]
                sq_7 = prefix_sq[right] - prefix_sq[left_7]
                mean_7 = np.divide(sum_7, count_7, out=np.zeros_like(sum_7), where=count_7 > 0)
                var_7 = np.divide(sq_7, count_7, out=np.zeros_like(sq_7), where=count_7 > 1) - np.square(mean_7)
                var_7[count_7 <= 1] = 0.0
                std_7 = np.sqrt(np.clip(var_7, 0.0, None))
                amt_zscore_7d[idx] = ((amts - mean_7) / (std_7 + 1.0)).astype(np.float32)

                seen_device = set()
                for i, d in enumerate(devs):
                    if d in seen_device:
                        is_new_device[idx[i]] = 0.0
                    else:
                        is_new_device[idx[i]] = 1.0
                        seen_device.add(d)

            row_order = work["row_id"].to_numpy(dtype=np.int64, copy=False)
            txn_count_1d_orig = np.zeros(len(df), dtype=np.float32)
            amt_zscore_7d_orig = np.zeros(len(df), dtype=np.float32)
            is_new_device_orig = np.zeros(len(df), dtype=np.float32)
            txn_count_1d_orig[row_order] = txn_count_1d
            amt_zscore_7d_orig[row_order] = amt_zscore_7d
            is_new_device_orig[row_order] = is_new_device

            # Add derived columns in one shot to avoid DataFrame fragmentation.
            derived = pd.DataFrame(
                {
                    "hour_sin": np.sin(2 * np.pi * hour / 24.0),
                    "hour_cos": np.cos(2 * np.pi * hour / 24.0),
                    "log_amount": np.log1p(pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)),
                    "txn_count_1d": txn_count_1d_orig,
                    "amt_zscore_7d": amt_zscore_7d_orig,
                    "is_new_device": is_new_device_orig,
                },
                index=df.index,
            )
            df = pd.concat([df, derived], axis=1)
            return df

        # Split is needed for leakage-safe imputations/statistics.
        t = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        q70 = np.quantile(t, 0.7)
        q85 = np.quantile(t, 0.85)
        df["split"] = np.where(t < q70, "train", np.where(t < q85, "val", "test"))

        # Card proxy identity and canonical IDs.
        df["card_id"] = df[self.CARD_COLS].fillna("NA").astype(str).agg("-".join, axis=1)
        df["email_id"] = (
            df["P_emaildomain"].fillna("NA").astype(str)
            + "|"
            + df["R_emaildomain"].fillna("NA").astype(str)
        )
        df["device_id"] = df["DeviceInfo"].fillna("unknown").astype(str)
        df["addr_id"] = df["addr1"].fillna("NA").astype(str)

        # Fill categorical fields consistently.
        cat_cols = set(self.TXN_CAT_COLS + ["DeviceInfo", "ProductCD", "P_emaildomain", "R_emaildomain"])
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)

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
                s = pd.to_numeric(df[col], errors="coerce")
                train_median = float(s[df["split"] == "train"].median()) if (df["split"] == "train").any() else float(s.median())
                if np.isnan(train_median):
                    train_median = 0.0
                df[col] = s.fillna(train_median)

        # Keep chronological order for rolling feature construction.
        df = df.sort_values(["card_id", "TransactionDT", "TransactionID"]).reset_index(drop=False)
        df = df.rename(columns={"index": "_orig_index"})

        n_rows = len(df)
        one_day = 86400.0
        seven_days = 7.0 * one_day
        fourteen_days = 14.0 * one_day
        thirty_days = 30.0 * one_day

        # Cardholder behavior profile.
        time_since_last = np.zeros(n_rows, dtype=np.float32)
        txn_count_1d = np.zeros(n_rows, dtype=np.float32)
        txn_count_3d = np.zeros(n_rows, dtype=np.float32)
        txn_count_7d = np.zeros(n_rows, dtype=np.float32)
        txn_count_14d = np.zeros(n_rows, dtype=np.float32)
        txn_count_30d = np.zeros(n_rows, dtype=np.float32)
        amt_mean_7d = np.zeros(n_rows, dtype=np.float32)
        amt_std_7d = np.zeros(n_rows, dtype=np.float32)
        amt_max_30d = np.zeros(n_rows, dtype=np.float32)
        unique_merch_30d = np.zeros(n_rows, dtype=np.float32)
        unique_device_30d = np.zeros(n_rows, dtype=np.float32)
        card_age_days = np.zeros(n_rows, dtype=np.float32)
        card_txn_seq = np.zeros(n_rows, dtype=np.float32)
        is_new_merchant = np.zeros(n_rows, dtype=np.float32)
        is_new_device = np.zeros(n_rows, dtype=np.float32)
        device_first_seen = np.zeros(n_rows, dtype=np.float32)
        addr_changed = np.zeros(n_rows, dtype=np.float32)

        for _, group in df.groupby("card_id", sort=False):
            idx = group.index.to_numpy()
            times = group["TransactionDT"].to_numpy(dtype=np.float64, copy=False)
            amts = group["TransactionAmt"].to_numpy(dtype=np.float64, copy=False)
            merch = group["ProductCD"].astype(str).to_numpy()
            devs = group["device_id"].astype(str).to_numpy()
            addrs = group["addr_id"].astype(str).to_numpy()

            n_group = len(group)
            if n_group == 0:
                continue

            # Sequence position and card age.
            card_txn_seq[idx] = np.arange(n_group, dtype=np.float32)
            card_age_days[idx] = ((times - times[0]) / one_day).astype(np.float32)

            if n_group > 1:
                time_since_last[idx[1:]] = np.diff(times).astype(np.float32, copy=False)

            starts_1d = np.searchsorted(times, times - one_day, side="left")
            starts_3d = np.searchsorted(times, times - 3.0 * one_day, side="left")
            starts_7d = np.searchsorted(times, times - seven_days, side="left")
            starts_14d = np.searchsorted(times, times - fourteen_days, side="left")
            starts_30d = np.searchsorted(times, times - thirty_days, side="left")
            positions = np.arange(n_group, dtype=np.float32)

            txn_count_1d[idx] = positions - starts_1d.astype(np.float32)
            txn_count_3d[idx] = positions - starts_3d.astype(np.float32)
            txn_count_7d[idx] = positions - starts_7d.astype(np.float32)
            txn_count_14d[idx] = positions - starts_14d.astype(np.float32)
            txn_count_30d[idx] = positions - starts_30d.astype(np.float32)

            prefix_sum = np.concatenate([[0.0], np.cumsum(amts, dtype=np.float64)])
            prefix_sq = np.concatenate([[0.0], np.cumsum(amts * amts, dtype=np.float64)])

            right = np.arange(1, n_group + 1)
            left_7 = starts_7d.astype(np.int64)
            count_7 = right - left_7
            sum_7 = prefix_sum[right] - prefix_sum[left_7]
            sq_7 = prefix_sq[right] - prefix_sq[left_7]
            mean_7 = np.divide(sum_7, count_7, out=np.zeros_like(sum_7), where=count_7 > 0)
            var_7 = np.divide(sq_7, count_7, out=np.zeros_like(sq_7), where=count_7 > 1) - np.square(mean_7)
            var_7[count_7 <= 1] = 0.0
            amt_mean_7d[idx] = mean_7.astype(np.float32)
            amt_std_7d[idx] = np.sqrt(np.clip(var_7, 0.0, None)).astype(np.float32)

            left_30 = starts_30d.astype(np.int64)
            max_30 = np.zeros(n_group, dtype=np.float32)
            uniq_merch = np.zeros(n_group, dtype=np.float32)
            uniq_devs = np.zeros(n_group, dtype=np.float32)

            for i in range(n_group):
                s = left_30[i]
                e = i
                if e <= s:
                    is_new_merchant[idx[i]] = 1.0
                    is_new_device[idx[i]] = 1.0
                    max_30[i] = 0.0
                    uniq_merch[i] = 0.0
                    uniq_devs[i] = 0.0
                    continue

                win_amts = amts[s:e]
                win_merch = merch[s:e]
                win_devs = devs[s:e]
                max_30[i] = float(np.max(win_amts)) if win_amts.size else 0.0
                uniq_merch[i] = float(len(set(win_merch)))
                uniq_devs[i] = float(len(set(win_devs)))
                is_new_merchant[idx[i]] = float(merch[i] not in set(win_merch))
                is_new_device[idx[i]] = float(devs[i] not in set(win_devs))

            amt_max_30d[idx] = max_30
            unique_merch_30d[idx] = uniq_merch
            unique_device_30d[idx] = uniq_devs

            first_addr = addrs[0]
            addr_changed[idx] = (addrs != first_addr).astype(np.float32)

            seen_device = set()
            for i, d in enumerate(devs):
                if d in seen_device:
                    device_first_seen[idx[i]] = 0.0
                else:
                    device_first_seen[idx[i]] = 1.0
                    seen_device.add(d)

        df["time_since_last_txn"] = time_since_last
        df["txn_count_1d"] = txn_count_1d
        df["txn_count_3d"] = txn_count_3d
        df["txn_count_7d"] = txn_count_7d
        df["txn_count_14d"] = txn_count_14d
        df["txn_count_30d"] = txn_count_30d
        df["amt_mean_7d"] = amt_mean_7d
        df["amt_std_7d"] = amt_std_7d
        df["amt_max_30d"] = amt_max_30d
        df["unique_merch_30d"] = unique_merch_30d
        df["unique_device_30d"] = unique_device_30d
        df["is_new_merchant"] = is_new_merchant
        df["is_new_device"] = is_new_device
        df["card_age_days"] = card_age_days
        df["card_txn_seq"] = card_txn_seq
        df["is_first_5_txns"] = (df["card_txn_seq"] < 5).astype(np.float32)

        # Deviation and velocity features.
        df["amt_zscore_7d"] = (df["TransactionAmt"] - df["amt_mean_7d"]) / (df["amt_std_7d"] + 1.0)
        df["amt_to_max_ratio"] = df["TransactionAmt"] / (df["amt_max_30d"] + 1.0)
        df["vel_1d_7d"] = df["txn_count_1d"] / (df["txn_count_7d"] + 1.0)
        df["vel_7d_30d"] = df["txn_count_7d"] / (df["txn_count_30d"] + 1.0)
        df["vel_surge"] = df["txn_count_1d"] / (df["txn_count_30d"] / 30.0 + 1.0)

        # Identity & trust features.
        df["email_mismatch"] = (df["P_emaildomain"] != df["R_emaildomain"]).astype(np.float32)
        df["p_email_is_free"] = df["P_emaildomain"].isin(self.FREE_EMAIL_DOMAINS).astype(np.float32)
        df["r_email_is_free"] = df["R_emaildomain"].isin(self.FREE_EMAIL_DOMAINS).astype(np.float32)
        df["both_emails_free"] = (df["p_email_is_free"] * df["r_email_is_free"]).astype(np.float32)

        m_match_cols = []
        for i in range(1, 10):
            col = f"M{i}"
            if col in df.columns:
                df[f"{col}_match"] = (df[col] == "T").astype(np.float32)
                df[f"{col}_known"] = df[col].notna().astype(np.float32)
                m_match_cols.append(f"{col}_match")
        if m_match_cols:
            df["m_match_score"] = df[m_match_cols].sum(axis=1)
        else:
            df["m_match_score"] = 0.0

        df["addr_changed"] = addr_changed
        df["device_first_seen"] = device_first_seen

        # Transaction context features.
        df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.float32)
        df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.float32)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.float32)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(np.float32)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["log_amount"] = np.log1p(df["TransactionAmt"])
        df["amount_cents"] = df["TransactionAmt"] % 1.0
        df["is_round"] = (df["amount_cents"] < 0.01).astype(np.float32)

        train_high_amt = float(df.loc[df["split"] == "train", "TransactionAmt"].quantile(0.9))
        if np.isnan(train_high_amt):
            train_high_amt = float(df["TransactionAmt"].quantile(0.9))
        df["night_highamt"] = df["is_night"] * (df["TransactionAmt"] > train_high_amt).astype(np.float32)
        df["weekend_newdev"] = df["is_weekend"] * df["is_new_device"]

        # Leakage-safe V-feature median imputation using train partition only.
        for col in self.V_COLS[: self.max_v_features]:
            if col in df.columns:
                median_val = df.loc[df["split"] == "train", col].median()
                if pd.isna(median_val):
                    median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(float(median_val))

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
        if self.feature_mode == "original":
            txn_feat_cols = [
                "log_amount",
                "hour_sin",
                "hour_cos",
                "dist1",
                "dist2",
                "txn_count_1d",
                "amt_zscore_7d",
                "is_new_device",
            ] + [c for c in self.V_COLS[: self.max_v_features] if c in df.columns]
        else:
            txn_feat_cols = [
                "log_amount",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_night",
                "is_round",
                "amount_cents",
                "time_since_last_txn",
                "txn_count_1d",
                "txn_count_3d",
                "txn_count_7d",
                "txn_count_14d",
                "txn_count_30d",
                "amt_mean_7d",
                "amt_std_7d",
                "amt_zscore_7d",
                "amt_max_30d",
                "amt_to_max_ratio",
                "unique_merch_30d",
                "unique_device_30d",
                "is_new_merchant",
                "is_new_device",
                "vel_1d_7d",
                "vel_7d_30d",
                "vel_surge",
                "card_age_days",
                "card_txn_seq",
                "is_first_5_txns",
                "email_mismatch",
                "p_email_is_free",
                "r_email_is_free",
                "both_emails_free",
                "m_match_score",
                "addr_changed",
                "device_first_seen",
                "night_highamt",
                "weekend_newdev",
            ] + [c for c in self.V_COLS[: self.max_v_features] if c in df.columns]
        txn_feats = df[txn_feat_cols].fillna(0).to_numpy(dtype=np.float32, copy=False)

        # Train-only standardization keeps values in a stable numeric range.
        train_rows = (df["split"].to_numpy() == "train")
        if train_rows.any():
            train_view = txn_feats[train_rows]
            train_mean = np.nanmean(train_view, axis=0)
            train_std = np.nanstd(train_view, axis=0)
            train_mean = np.nan_to_num(train_mean, nan=0.0, posinf=0.0, neginf=0.0)
            train_std = np.nan_to_num(train_std, nan=1.0, posinf=1.0, neginf=1.0)
            train_std = np.where(train_std < 1e-3, 1.0, train_std)
            txn_feats = (txn_feats - train_mean) / train_std

        # Final sanitization: remove non-finite values and clamp outliers.
        txn_feats = np.nan_to_num(txn_feats, nan=0.0, posinf=0.0, neginf=0.0)
        txn_feats = np.clip(txn_feats, -10.0, 10.0).astype(np.float32, copy=False)
        data["txn"].x = torch.from_numpy(txn_feats)

        # Build compact per-transaction history indices to avoid dense [N, K, D] memory.
        seq_index = np.full((len(df), self.seq_len), -1, dtype=np.int32)
        # Keep temporal deltas in float32 to avoid overflow on large gaps.
        delta_t = np.zeros((len(df), self.seq_len), dtype=np.float32)
        seq_mask = np.ones((len(df), self.seq_len), dtype=bool)

        df_sorted = df[["card_id", "TransactionDT", "TransactionID"]].copy()
        df_sorted["row_id"] = np.arange(len(df))
        df_sorted = df_sorted.sort_values(["card_id", "TransactionDT", "TransactionID"]).reset_index(drop=True)

        for _, group in df_sorted.groupby("card_id", sort=False):
            rows = group["row_id"].to_numpy(dtype=np.int32, copy=False)
            times = group["TransactionDT"].to_numpy(dtype=np.float64, copy=False)
            for i, row_id in enumerate(rows):
                start = max(0, i - self.seq_len)
                hist_rows = rows[start:i]
                if hist_rows.size == 0:
                    continue
                k = hist_rows.size
                seq_index[row_id, -k:] = hist_rows
                raw_delta = (times[i] - times[start:i]).astype(np.float32)
                # Compress dynamic range: convert to hours and apply log1p.
                delta = np.log1p(np.clip(raw_delta, 0.0, None) / 3600.0)
                delta_t[row_id, -k:] = delta
                seq_mask[row_id, -k:] = False

        data["txn"].seq_index = torch.from_numpy(seq_index)
        data["txn"].delta_t = torch.from_numpy(delta_t)
        data["txn"].seq_mask = torch.from_numpy(seq_mask)

        # ── Training set for aggregates (prevent leakage) ──
        train_df = df[df["split"] == "train"]

        # ── Card features (aggregate per card) ──
        card_feat = train_df.groupby("card_id").agg(
            avg_amount=("TransactionAmt", "mean"),
            txn_count=("TransactionID", "count"),
            std_amount=("TransactionAmt", "std"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(card_map).fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["card"].x = torch.from_numpy(card_feat)

        # ── Merchant features ──
        merch_feat = train_df.groupby("ProductCD").agg(
            avg_amount=("TransactionAmt", "mean"),
            txn_count=("TransactionID", "count"),
            fraud_rate=("isFraud", "mean"),
        ).fillna(0).reindex(merch_map).fillna(0).to_numpy(dtype=np.float32, copy=False)
        data["merchant"].x = torch.from_numpy(merch_feat)

        # ── Device features ──
        device_feat = train_df.groupby("DeviceInfo").agg(
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
        max_cards_per_device = self.max_cards_per_device
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
            train=data,
            val=data,
            test=data,
            metadata={
                "source": "ieee_cis",
                "n_txns": len(df),
                "fraud_rate": float(df["isFraud"].mean()),
                "n_card_nodes": data["card"].x.shape[0],
                "n_merchant_nodes": data["merchant"].x.shape[0],
                "max_v_features": self.max_v_features,
                "max_cards_per_device": self.max_cards_per_device,
                "feature_mode": self.feature_mode,
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

        return DatasetSplit(train=data, val=data, test=data, metadata={"source": "elliptic", "n_labeled": len(labeled)})


# ─────────────────────────────────────────────────────────────────
# UNIFIED PIPELINE
# ─────────────────────────────────────────────────────────────────

def load_all_datasets(
    data_dir: str,
    datasets: List[str] = None,
    ieee_max_rows: Optional[int] = None,
    ieee_max_v_features: int = 160,
    ieee_max_cards_per_device: int = 20,
    ieee_feature_mode: str = "enhanced",
) -> Dict[str, DatasetSplit]:
    """Load and return all available datasets."""
    if datasets is None:
        datasets = ["ieee_cis", "paysim", "elliptic"]

    results = {}

    pipeline_map = {
        "ieee_cis": lambda: IEEECISPipeline(
            f"{data_dir}/ieee_cis",
            max_rows=ieee_max_rows,
            max_v_features=ieee_max_v_features,
            max_cards_per_device=ieee_max_cards_per_device,
            feature_mode=ieee_feature_mode,
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
