"""資料驗證、清理與特徵工程（FR-01～FR-04）。

與 templates/*.html 內的 JS `toFeat()` 必須逐欄一致；任何規則變更需同步兩邊並重跑 TC-05/TC-06。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "撥貸金額"

CATS: dict[str, dict] = {
    "行業": {"opts": ["行業屬性A", "行業屬性B", "行業屬性C", "行業屬性D", "行業屬性E"], "unk": "行業_未知"},
    "學歷": {"opts": ["高中", "大學", "博士", "專科", "碩士"], "unk": "學歷_未知"},
    "居住": {"opts": ["住本人資產無貸款", "住公司宿舍", "住租賃", "住親戚產業", "住本人資產有貸款"], "unk": "居住_未知"},
    "婚姻": {"opts": ["婚姻已婚", "婚姻已婚無子女", "婚姻未婚", "婚姻已婚有子女", "婚姻其他"], "unk": "婚姻_未知"},
}
NUM_COLS = ["循環比", "信用卡循環金額", "無擔保總餘額", "預借現金次數", "信用貸款含現金卡家數", "DBR"]
CAT_COLS = [c for g in CATS.values() for c in g["opts"]]
RAW_COLS = CAT_COLS + NUM_COLS  # 26 個原始特徵，順序同 regression8.csv
UNK_COLS = [g["unk"] for g in CATS.values()]
MODEL_COLS = RAW_COLS + UNK_COLS + ["DBR_為0"]  # 31 欄，順序即模型輸入順序

DBR_Q = (0.01, 0.99)


class SchemaError(ValueError):
    """輸入檔缺少必要欄位。"""


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """FR-01：缺必要欄位時拋 SchemaError，列出缺少的欄位名。"""
    need = (RAW_COLS + [TARGET]) if require_target else NUM_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SchemaError("缺少欄位：" + "、".join(missing))


def clean_training(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FR-02：移除完全重複列、目標缺漏或 ≤0 的列。回傳 (乾淨資料, 統計)。"""
    validate_schema(df, require_target=True)
    stats = {"rows_in": int(len(df))}
    df = df.drop_duplicates()
    stats["duplicates_removed"] = stats["rows_in"] - int(len(df))
    y = pd.to_numeric(df[TARGET], errors="coerce")
    ok = y.notna() & (y > 0)
    stats["invalid_target_removed"] = int((~ok).sum())
    df = df.loc[ok].copy()
    df[TARGET] = y[ok].astype(float)
    stats["rows_out"] = int(len(df))
    return df.reset_index(drop=True), stats


def fit_dbr_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """FR-03：DBR 截斷界線 = 訓練資料 1%、99% 分位數（取 2 位小數）。"""
    q = pd.to_numeric(df["DBR"], errors="coerce").fillna(0).quantile(list(DBR_Q))
    return round(float(q.iloc[0]), 2), round(float(q.iloc[1]), 2)


def make_features(df: pd.DataFrame, dbr_bounds: tuple[float, float]) -> pd.DataFrame:
    """FR-04：產生 31 欄模型特徵。規則（與 JS toFeat 相同）：

    - 類別欄：值 == 1 視為 1，其他（空白、非數字、0）視為 0；
      同組 5 欄皆非 1 → 該組「_未知」= 1。
    - 數值欄：非數字或空白 → 0。
    - 無擔保總餘額 < 0 → 0。
    - DBR_為0 = 1 當原始 DBR == 0（在截斷前判斷）。
    - DBR 截斷到 dbr_bounds。
    """
    out = pd.DataFrame(index=df.index)
    for g in CATS.values():
        hit = np.zeros(len(df), dtype=bool)
        for o in g["opts"]:
            v = (pd.to_numeric(df[o], errors="coerce") == 1) if o in df.columns else pd.Series(False, index=df.index)
            out[o] = v.astype(float).to_numpy()
            hit |= v.to_numpy()
        out[g["unk"]] = (~hit).astype(float)
    for c in NUM_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    out["無擔保總餘額"] = out["無擔保總餘額"].clip(lower=0)
    out["DBR_為0"] = (out["DBR"] == 0).astype(float)
    out["DBR"] = out["DBR"].clip(dbr_bounds[0], dbr_bounds[1])
    return out[MODEL_COLS]


def level(amount: float, cut: tuple[float, float]) -> str:
    """FR-07：預測等級。≤cut0 低；≤cut1 中；其餘 高。"""
    return "低" if amount <= cut[0] else ("中" if amount <= cut[1] else "高")
