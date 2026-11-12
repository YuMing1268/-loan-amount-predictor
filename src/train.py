"""訓練、評估與匯出（FR-05、FR-06、FR-09、NFR-01、NFR-03）。

用法：
    python src/train.py --data regression8.csv --out .

產出：
    models/hgb.json、models/tree.json      模型參數（瀏覽器可執行的格式）
    reports/metrics.json、reports/metrics.md  保留集評估、品質門檻、消融實驗
    dist/撥貸金額預測工具.html、dist/撥貸金額(決策樹判斷).html
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (MODEL_COLS, TARGET, clean_training, fit_dbr_bounds,  # noqa: E402
                      make_features)

SEED = 42
TEST_SIZE = 0.2
HGB_PARAMS = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
                  early_stopping=False, random_state=SEED)
TREE_PARAMS = dict(max_depth=4, min_samples_leaf=200, random_state=SEED)
# 品質門檻（NFR-01）：事先訂定，不依結果調整
GATE_HGB_VS_BASE = 0.85  # HGB 保留集 MAE ≤ 基準 MAE × 0.85
GATE_TREE_VS_BASE = 0.95  # 決策樹 保留集 MAE ≤ 基準 MAE × 0.95
LEAK_COLS = ["無擔保總餘額", "DBR", "DBR_為0"]  # brief 主要約束：須為撥貸前資料


# ---------- 模型 ----------
def fit_hgb(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(X.to_numpy(), np.log1p(y.to_numpy()))
    return m


def predict_hgb(m, X: pd.DataFrame) -> np.ndarray:
    return np.maximum(0, np.expm1(m.predict(X.to_numpy())))


def fit_tree(X: pd.DataFrame, y: pd.Series) -> DecisionTreeRegressor:
    m = DecisionTreeRegressor(**TREE_PARAMS)
    m.fit(X.to_numpy(), y.to_numpy())
    return m


def tree_leaf_medians(m, X: pd.DataFrame, y: pd.Series) -> dict[int, float]:
    leaf = m.apply(X.to_numpy())
    return pd.Series(y.to_numpy()).groupby(leaf).median().to_dict()


def predict_tree_median(m, med: dict[int, float], X: pd.DataFrame) -> np.ndarray:
    return np.array([med[i] for i in m.apply(X.to_numpy())])


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": round(float(mean_absolute_error(y, p)), 3),
        "MedianAE": round(float(np.median(np.abs(y - p))), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
        "R2": round(float(r2_score(y, p)), 4),
        "within_30pct": round(float(np.mean(np.abs(y - p) <= 0.3 * y)), 4),
    }


# ---------- 匯出 ----------
def export_hgb(m, dbr, median, cut, n) -> dict:
    trees = []
    for it in m._predictors:
        nodes = it[0].nodes
        trees.append({
            "f": [int(x) if not leaf else -1 for x, leaf in zip(nodes["feature_idx"], nodes["is_leaf"])],
            "t": [float(x) for x in nodes["num_threshold"]],
            "l": [int(x) for x in nodes["left"]],
            "r": [int(x) for x in nodes["right"]],
            "v": [float(f"{x:.10g}") for x in nodes["value"]],
        })
    return {"base": float(np.ravel(m._baseline_prediction)[0]), "trees": trees, "cols": MODEL_COLS,
            "dbr": list(dbr), "median": median, "cut": list(cut), "log": True, "n": n}


def export_tree(m, X: pd.DataFrame, y: pd.Series, dbr, median, n) -> dict:
    t = m.tree_
    leaf_of = m.apply(X.to_numpy())
    yv = y.to_numpy()

    def node(i):
        if t.children_left[i] == -1:
            v = yv[leaf_of == i]
            return {"n": int(v.size), "med": float(np.median(v)), "mean": round(float(v.mean()), 1),
                    "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
                    "hi": round(float(np.mean(v > median)), 3)}
        return {"f": MODEL_COLS[t.feature[i]], "t": round(float(t.threshold[i]), 6),
                "l": node(t.children_left[i]), "r": node(t.children_right[i])}

    return {"tree": node(0), "median": median, "dbr": list(dbr), "n": n, "cols": MODEL_COLS}


def build_html(template: Path, model: dict, out: Path, trained_at: str) -> None:
    src = template.read_text(encoding="utf-8")
    assert "/*__MODEL__*/null" in src, f"{template} 缺少 /*__MODEL__*/null 置換點"
    model = dict(model, trained_at=trained_at)
    js = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    out.write_text(src.replace("/*__MODEL__*/null", js), encoding="utf-8", newline="\n")


# ---------- 主流程 ----------
def run(data: Path, root: Path, skip_ablation: bool = False) -> dict:
    t0 = time.time()
    raw = pd.read_csv(data, encoding="utf-8-sig")
    df, clean_stats = clean_training(raw)
    y = df[TARGET]
    print(f"[data] {clean_stats}")

    # 1) 保留集評估：界線只由訓練分割估計，避免資訊外洩
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    dbr_tr = fit_dbr_bounds(tr)
    Xtr, Xte = make_features(tr, dbr_tr), make_features(te, dbr_tr)
    ytr, yte = tr[TARGET], te[TARGET]
    base_med = float(ytr.median())

    hgb = fit_hgb(Xtr, ytr)
    tree = fit_tree(Xtr, ytr)
    med = tree_leaf_medians(tree, Xtr, ytr)
    holdout = {
        "baseline_median": metrics(yte, np.full(len(yte), base_med)),
        "hgb": metrics(yte, predict_hgb(hgb, Xte)),
        "tree_leaf_median": metrics(yte, predict_tree_median(tree, med, Xte)),
    }
    print(f"[holdout] {holdout}")

    ablation = None
    if not skip_ablation:
        keep = [c for c in MODEL_COLS if c not in LEAK_COLS]
        m2 = HistGradientBoostingRegressor(**HGB_PARAMS).fit(Xtr[keep].to_numpy(), np.log1p(ytr.to_numpy()))
        ablation = {"removed": LEAK_COLS,
                    "hgb_without": metrics(yte, np.maximum(0, np.expm1(m2.predict(Xte[keep].to_numpy()))))}
        print(f"[ablation] {ablation}")

    bmae = holdout["baseline_median"]["MAE"]
    gates = {
        "hgb_mae_le_0.85_baseline": holdout["hgb"]["MAE"] <= GATE_HGB_VS_BASE * bmae,
        "tree_mae_le_0.95_baseline": holdout["tree_leaf_median"]["MAE"] <= GATE_TREE_VS_BASE * bmae,
    }

    # 2) 最終模型：全部乾淨資料重訓後匯出
    dbr = fit_dbr_bounds(df)
    X = make_features(df, dbr)
    median = float(y.median())
    hgb_f = fit_hgb(X, y)
    p_all = predict_hgb(hgb_f, X)
    cut = (float(np.quantile(p_all, 1 / 3)), float(np.quantile(p_all, 2 / 3)))
    tree_f = fit_tree(X, y)
    n = int(len(df))
    hgb_json = export_hgb(hgb_f, dbr, median, cut, n)
    tree_json = export_tree(tree_f, X, y, dbr, median, n)

    # 匯出前自我核對：JSON 參數重算 == sklearn（防止匯出格式錯誤）
    chk = X.sample(2000, random_state=SEED)
    assert np.allclose(predict_from_json(hgb_json, chk), predict_hgb(hgb_f, chk), rtol=1e-6, atol=1e-6)

    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    for d in ("models", "reports", "dist"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "models/hgb.json").write_text(json.dumps(hgb_json, ensure_ascii=False), encoding="utf-8")
    (root / "models/tree.json").write_text(json.dumps(tree_json, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = Path(__file__).resolve().parent.parent / "templates"
    if all(gates.values()):  # FR-09：門檻未過不覆蓋 dist，避免較差模型上線
        build_html(tpl / "predictor.html", hgb_json, root / "dist/撥貸金額預測工具.html", trained_at)
        build_html(tpl / "tree.html", tree_json, root / "dist/撥貸金額(決策樹判斷).html", trained_at)
    else:
        print("[gates] 未通過，dist/ 保持不變")

    report = {"trained_at": trained_at, "data": str(data), "clean": clean_stats, "seed": SEED,
              "test_size": TEST_SIZE, "hgb_params": HGB_PARAMS, "tree_params": TREE_PARAMS,
              "dbr_bounds_final": dbr, "median_final": median, "cut_final": cut,
              "holdout": holdout, "ablation": ablation, "gates": gates,
              "seconds": round(time.time() - t0, 1)}
    (root / "reports/metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "reports/metrics.md").write_text(render_md(report), encoding="utf-8")
    print(f"[gates] {gates}  ({report['seconds']}s)")
    return report


def predict_from_json(M: dict, X: pd.DataFrame) -> np.ndarray:
    """以匯出的 JSON 參數逐樹推論（與 JS predict 相同演算法）。"""
    x = X[M["cols"]].to_numpy()
    s = np.full(len(x), M["base"])
    for T in M["trees"]:
        f, t, l, r, v = (np.array(T[k]) for k in "ftlrv")
        idx = np.zeros(len(x), dtype=int)
        while True:
            live = f[idx] >= 0
            if not live.any():
                break
            go_left = x[np.arange(len(x)), np.where(live, f[idx], 0)] <= t[idx]
            idx = np.where(live, np.where(go_left, l[idx], r[idx]), idx)
        s += v[idx]
    return np.maximum(0, np.expm1(s))


def render_md(r: dict) -> str:
    h = r["holdout"]
    rows = "\n".join(f"| {k} | {v['MAE']} | {v['MedianAE']} | {v['RMSE']} | {v['R2']} | {v['within_30pct']:.1%} |"
                     for k, v in h.items())
    abl = ""
    if r["ablation"]:
        a = r["ablation"]["hgb_without"]
        abl = (f"\n## 消融：移除 {'、'.join(r['ablation']['removed'])}\n\n"
               f"| 模型 | MAE | R2 |\n|---|---|---|\n| HGB 全特徵 | {h['hgb']['MAE']} | {h['hgb']['R2']} |\n"
               f"| HGB 移除後 | {a['MAE']} | {a['R2']} |\n\n"
               "若這些欄位實際為撥貸後資料，上線成績會接近「移除後」一列（brief 主要約束）。\n")
    gates = "\n".join(f"- {k}：{'通過' if v else '**未通過**'}" for k, v in r["gates"].items())
    return (f"# 模型評估報告\n\n訓練時間 {r['trained_at']}；資料 `{r['data']}`；清理 {r['clean']}。\n\n"
            f"保留集 {int(r['test_size']*100)}%（random_state={r['seed']}）。\n\n"
            f"| 模型 | MAE | MedianAE | RMSE | R2 | 誤差≤30% |\n|---|---|---|---|---|---|\n{rows}\n"
            f"{abl}\n## 品質門檻（NFR-01）\n\n{gates}\n\n"
            f"## 匯出參數\n\nDBR 截斷 {r['dbr_bounds_final']}；全體中位數 {r['median_final']}；"
            f"等級切點 {[round(c, 2) for c in r['cut_final']]}。\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="regression8.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    rep = run(Path(a.data), Path(a.out), a.skip_ablation)
    sys.exit(0 if all(rep["gates"].values()) else 2)
