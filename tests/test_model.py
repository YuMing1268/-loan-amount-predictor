"""TC-05～TC-08：模型品質、瀏覽器推論一致性、可重現性。"""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import DATA, PRED_HTML, TREE_HTML, node
from features import RAW_COLS, TARGET, fit_dbr_bounds, make_features
import train


@pytest.fixture(scope="module")
def sample(clean_df):
    return clean_df.sample(1500, random_state=7)


@pytest.mark.req("TC-05", "AC-01.1", "FR-06", "NFR-02")
def test_js_predictor_matches_sklearn(artifacts, clean_df, sample):
    """獨立重訓 sklearn 模型當 oracle，與 dist 預測工具的 JS 推論逐筆比對。"""
    dbr = fit_dbr_bounds(clean_df)
    m = train.fit_hgb(make_features(clean_df, dbr), clean_df[TARGET])
    want = train.predict_hgb(m, make_features(sample, dbr))
    recs = sample[RAW_COLS].to_dict("records")
    got = np.array([a for a, _ in node(PRED_HTML, "predict", recs)])
    assert np.max(np.abs(got - want)) < 1e-3, "JS 與 sklearn 預測不一致"


@pytest.mark.req("TC-06", "AC-02.1", "FR-08", "NFR-02")
def test_js_tree_matches_sklearn(artifacts, clean_df, sample):
    dbr = fit_dbr_bounds(clean_df)
    X = make_features(clean_df, dbr)
    m = train.fit_tree(X, clean_df[TARGET])
    med = train.tree_leaf_medians(m, X, clean_df[TARGET])
    leaf = m.apply(make_features(sample, dbr).to_numpy())
    got = node(TREE_HTML, "judge", sample[RAW_COLS].to_dict("records"))
    assert [g["med"] for g in got] == [med[i] for i in leaf]
    assert all(1 <= g["steps"] <= 4 for g in got)  # 深度 4 → 路徑最多 4 步
    meta = node(TREE_HTML, "meta")
    assert all(g["hi"] == (g["med"] > meta["median"]) for g in got)


@pytest.mark.req("TC-06", "FR-08")
def test_tree_split_boundary_goes_left(artifacts):
    """SRS 規則：特徵值 == 門檻 走左（≤），略大於門檻走右。真實資料不會剛好落在門檻上，故另測。"""
    tree = json.loads((DATA.parent / "models/tree.json").read_text(encoding="utf-8"))["tree"]
    cols = node(TREE_HTML, "meta")["cols"]
    base = {c: 0 for c in cols}
    eq, gt = dict(base), dict(base)
    eq[tree["f"]] = tree["t"]
    gt[tree["f"]] = tree["t"] + 1e-6
    assert [p[0] for p in node(TREE_HTML, "walk", [eq, gt])] == [True, False]


@pytest.mark.req("TC-07", "NFR-01", "NFR-03")
def test_quality_gates_and_ablation(artifacts):
    h = artifacts["holdout"]
    assert all(artifacts["gates"].values()), artifacts["gates"]
    assert h["hgb"]["MAE"] < h["tree_leaf_median"]["MAE"] < h["baseline_median"]["MAE"]
    assert artifacts["clean"]["rows_out"] == node(PRED_HTML, "meta")["n"]
    ab = artifacts["ablation"]
    assert ab and set(ab["removed"]) == {"無擔保總餘額", "DBR", "DBR_為0"}
    assert ab["hgb_without"]["MAE"] > h["hgb"]["MAE"]  # 量化對兩欄位的依賴（brief 主要約束）


@pytest.mark.req("TC-08", "FR-09", "NFR-04")
def test_training_is_reproducible(tmp_path):
    small = pd.read_csv(DATA, encoding="utf-8-sig", nrows=4000)
    f = tmp_path / "small.csv"
    small.to_csv(f, index=False, encoding="utf-8-sig")
    outs = []
    for k in ("a", "b"):
        train.run(f, tmp_path / k, skip_ablation=True)
        outs.append(json.loads((tmp_path / k / "models/hgb.json").read_text(encoding="utf-8")))
        assert (tmp_path / k / "dist/撥貸金額預測工具.html").exists()
    assert outs[0] == outs[1]


@pytest.mark.req("TC-08", "AC-04.2", "FR-09")
def test_gate_failure_keeps_dist_unchanged(tmp_path, monkeypatch):
    small = pd.read_csv(DATA, encoding="utf-8-sig", nrows=4000)
    f = tmp_path / "small.csv"
    small.to_csv(f, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(train, "GATE_HGB_VS_BASE", 0.01)  # 刻意設成不可能達到
    rep = train.run(f, tmp_path / "o", skip_ablation=True)
    assert rep["gates"]["hgb_mae_le_0.85_baseline"] is False
    assert (tmp_path / "o/reports/metrics.json").exists()
    assert not (tmp_path / "o/dist/撥貸金額預測工具.html").exists()
