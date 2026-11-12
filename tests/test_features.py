"""TC-01～TC-04：資料驗證、清理、特徵規則、等級切點（預期值由規格人工算出）。"""
import pandas as pd
import pytest

from conftest import PRED_HTML, node
from features import (MODEL_COLS, RAW_COLS, TARGET, SchemaError, clean_training,
                      level, make_features, validate_schema)

BOUNDS = (1.13, 21.98)


def row(**kw):
    r = {c: 0 for c in RAW_COLS}
    r.update(kw)
    return r


@pytest.mark.req("TC-01", "AC-03.2", "FR-01")
def test_schema_rejects_missing_columns():
    df = pd.DataFrame([row()]).drop(columns=["DBR", "循環比"])
    df[TARGET] = 10
    with pytest.raises(SchemaError) as e:
        validate_schema(df)
    assert "循環比" in str(e.value) and "DBR" in str(e.value)
    # 預測用途不需目標欄，只需 6 個數值欄
    validate_schema(pd.DataFrame([row()]), require_target=False)


@pytest.mark.req("TC-02", "FR-02")
def test_clean_removes_duplicates_and_invalid_target():
    rows = [dict(row(DBR=10), **{TARGET: 50}), dict(row(DBR=10), **{TARGET: 50}),  # 完全重複 → 留 1
            dict(row(DBR=11), **{TARGET: 0}),  # 目標 0 → 移除
            dict(row(DBR=12), **{TARGET: "x"}),  # 目標非數字 → 移除
            dict(row(DBR=13), **{TARGET: 30})]
    df, st = clean_training(pd.DataFrame(rows))
    assert st == {"rows_in": 5, "duplicates_removed": 1, "invalid_target_removed": 2, "rows_out": 2}
    assert sorted(df[TARGET].tolist()) == [30.0, 50.0]


CASES = [
    # (輸入, 預期非零特徵) — 依 SRS 規則手算
    (row(行業屬性C=1, 大學=1, 循環比="abc", 無擔保總餘額=-100, DBR=0),
     {"行業屬性C": 1, "大學": 1, "居住_未知": 1, "婚姻_未知": 1, "DBR_為0": 1, "DBR": 1.13}),
    (row(行業屬性A="1.0", 高中=2, 住租賃=1, 婚姻未婚=1, DBR=25, 信用卡循環金額=3.5),
     {"行業屬性A": 1, "學歷_未知": 1, "住租賃": 1, "婚姻未婚": 1, "DBR": 21.98, "信用卡循環金額": 3.5}),
    (row(行業屬性E=1, 碩士=1, 住公司宿舍=1, 婚姻其他=1, DBR=-5, 無擔保總餘額=20, 預借現金次數=3),
     {"行業屬性E": 1, "碩士": 1, "住公司宿舍": 1, "婚姻其他": 1, "DBR": 1.13, "無擔保總餘額": 20, "預借現金次數": 3}),
    (row(行業屬性B=1, 博士=1, 住親戚產業=1, 婚姻已婚=1, DBR=12.5, 循環比=""),
     {"行業屬性B": 1, "博士": 1, "住親戚產業": 1, "婚姻已婚": 1, "DBR": 12.5}),
]


@pytest.mark.req("TC-03", "FR-04")
@pytest.mark.parametrize("rec,expect", CASES)
def test_feature_rules_python(rec, expect):
    got = make_features(pd.DataFrame([rec]), BOUNDS).iloc[0].to_dict()
    want = {c: float(expect.get(c, 0)) for c in MODEL_COLS}
    assert got == pytest.approx(want)


@pytest.mark.req("TC-03", "FR-04", "NFR-02")
def test_feature_rules_js_matches_python(artifacts):
    """JS toFeat 與 Python make_features 對相同輸入給出相同 31 欄（使用成品的 DBR 界線）。"""
    meta = node(PRED_HTML, "meta")
    recs = [{k: (v if isinstance(v, str) else float(v)) for k, v in r.items()} for r, _ in CASES]
    js = node(PRED_HTML, "toFeat", recs)
    py = make_features(pd.DataFrame([r for r, _ in CASES]), tuple(meta["dbr"]))
    assert meta["cols"] == MODEL_COLS
    for i, f in enumerate(js):
        assert [f[c] for c in MODEL_COLS] == pytest.approx(py.iloc[i].tolist())


@pytest.mark.req("TC-04", "AC-01.2", "FR-07")
def test_level_boundaries(artifacts):
    assert [level(x, (30, 50)) for x in (0, 30, 30.01, 50, 50.01)] == ["低", "低", "中", "中", "高"]
    c0, c1 = node(PRED_HTML, "meta")["cut"]
    assert node(PRED_HTML, "lvl", [c0, c0 + 1e-9, c1, c1 + 1e-9]) == ["低", "中", "中", "高"]
