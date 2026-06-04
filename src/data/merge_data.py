"""Data merge logic for transactions, cards, users, MCC and labels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.config.settings import Settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake-like lowercase names."""
    output = df.copy()
    output.columns = (
        output.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return output


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate column that exists."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _normalize_label_value(value: Any) -> int:
    """Convert common fraud labels to 0/1."""
    if pd.isna(value):
        return 0
    normalized = str(value).strip().lower()
    return int(normalized in {"1", "true", "yes", "y", "fraud", "fraudulent"})


class FraudDataMerger:
    """Merge raw dataset components into one supervised transaction table."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _labels_to_frame(self, labels: object) -> pd.DataFrame:
        """Convert Kaggle label JSON variants into a normalized dataframe."""
        if isinstance(labels, dict) and "target" in labels and isinstance(labels["target"], dict):
            frame = pd.DataFrame(
                {
                    "transaction_id": list(labels["target"].keys()),
                    self.settings.target_column: list(labels["target"].values()),
                }
            )
        elif isinstance(labels, dict):
            frame = pd.DataFrame(
                {
                    "transaction_id": list(labels.keys()),
                    self.settings.target_column: list(labels.values()),
                }
            )
        elif isinstance(labels, list):
            frame = pd.DataFrame(labels)
        else:
            raise ValueError("Formato de labels nao suportado.")

        frame = normalize_columns(frame)
        id_col = first_existing(frame.columns, self.settings.transaction_id_candidates)
        if id_col is None:
            id_col = frame.columns[0]
        target_col = first_existing(frame.columns, (self.settings.target_column, "target", "fraud"))
        if target_col is None:
            target_col = frame.columns[-1]

        frame = frame[[id_col, target_col]].rename(
            columns={id_col: "transaction_id", target_col: self.settings.target_column}
        )
        frame["transaction_id"] = frame["transaction_id"].astype(str)
        frame[self.settings.target_column] = frame[self.settings.target_column].map(_normalize_label_value)
        return frame

    def _mcc_to_frame(self, mcc_codes: object) -> pd.DataFrame:
        """Convert MCC JSON into a dataframe with code and description."""
        if not mcc_codes:
            return pd.DataFrame()
        if isinstance(mcc_codes, dict):
            frame = pd.DataFrame({"mcc": list(mcc_codes.keys()), "mcc_description": list(mcc_codes.values())})
        elif isinstance(mcc_codes, list):
            frame = pd.DataFrame(mcc_codes)
            frame = normalize_columns(frame)
            code_col = first_existing(frame.columns, ("mcc", "code", "mcc_code"))
            desc_col = first_existing(frame.columns, ("description", "mcc_description", "name"))
            if code_col and desc_col:
                frame = frame[[code_col, desc_col]].rename(
                    columns={code_col: "mcc", desc_col: "mcc_description"}
                )
        else:
            return pd.DataFrame()
        frame["mcc"] = frame["mcc"].astype(str)
        return frame[["mcc", "mcc_description"]].drop_duplicates()

    def merge(
        self,
        transactions: pd.DataFrame,
        cards: pd.DataFrame,
        users: pd.DataFrame,
        mcc_codes: object,
        labels: object,
    ) -> pd.DataFrame:
        """Merge all raw components and return a supervised transaction dataframe."""
        tx = normalize_columns(transactions)
        tx_id_col = first_existing(tx.columns, self.settings.transaction_id_candidates)
        if tx_id_col is None:
            raise ValueError("Nao foi encontrada coluna de ID da transacao.")
        tx = tx.rename(columns={tx_id_col: "transaction_id"})
        tx["transaction_id"] = tx["transaction_id"].astype(str)

        merged = tx.merge(self._labels_to_frame(labels), on="transaction_id", how="inner")

        if not cards.empty:
            card_df = normalize_columns(cards)
            card_key = first_existing(card_df.columns, ("id", "card_id"))
            tx_card_key = first_existing(merged.columns, self.settings.card_id_candidates)
            if card_key and tx_card_key:
                card_df = card_df.rename(columns={card_key: tx_card_key})
                merged = merged.merge(card_df, on=tx_card_key, how="left", suffixes=("", "_card"))

        if not users.empty:
            user_df = normalize_columns(users)
            user_key = first_existing(user_df.columns, ("id", "client_id", "user_id", "customer_id"))
            tx_user_key = first_existing(merged.columns, self.settings.user_id_candidates)
            if user_key and tx_user_key:
                user_df = user_df.rename(columns={user_key: tx_user_key})
                merged = merged.merge(user_df, on=tx_user_key, how="left", suffixes=("", "_user"))

        mcc_frame = self._mcc_to_frame(mcc_codes)
        if not mcc_frame.empty and "mcc" in merged.columns:
            merged["mcc"] = merged["mcc"].astype(str)
            merged = merged.merge(mcc_frame, on="mcc", how="left")

        logger.info("Dataset consolidado: %s linhas, %s colunas", merged.shape[0], merged.shape[1])
        return merged
