"""TC-09～TC-12：批次 CSV、編碼、警示與每日檔名、畫面冒煙測試。"""
import base64
import json
import subprocess

from conftest import ROOT, TREE_HTML

import pytest

from conftest import PRED_HTML, node
from features import RAW_COLS

HDR = ",".join(RAW_COLS)


def line(**kw):
    v = {c: "0" for c in RAW_COLS}
    v.update({k: str(x) for k, x in kw.items()})
    return ",".join(v[c] for c in RAW_COLS)


@pytest.mark.req("TC-09", "AC-03.2", "FR-10")
@pytest.mark.parametrize("text,err", [
    (HDR.replace(",DBR", ""), "檔案沒有資料列。"),
    (HDR.replace(",DBR", "") + "\n" + line()[:-2], "缺少欄位：DBR"),
    ("", "檔案沒有資料列。"),
])
def test_batch_rejects(artifacts, text, err):
    assert node(PRED_HTML, "batch", text) == {"error": err}


@pytest.mark.req("TC-09", "AC-03.1", "FR-10")
def test_batch_output(artifacts):
    rows = [line(行業屬性C=1, 大學=1, DBR=12, 無擔保總餘額=20, 循環比=10),
            line(行業屬性A=1, DBR=18, 無擔保總餘額=10),
            line(行業屬性E=1, DBR=5)]
    amounts = [80, 120, 15]
    text = "﻿撥貸金額," + HDR + ",備註\r\n" + "\r\n".join(
        f'{a},{r},"含,逗號"' for a, r in zip(amounts, rows)) + "\r\n"
    out = node(PRED_HTML, "batch", text)
    recs = [dict(zip(RAW_COLS, r.split(","))) for r in rows]
    pred = node(PRED_HTML, "predict", recs)
    r1 = [round(a * 10) / 10 for a, _ in pred]
    lines = out["csv"].lstrip("﻿").split("\r\n")
    assert out["n"] == 3 and len(lines) == 4
    assert lines[0].endswith(",備註,預測撥貸金額,預測等級,預測高低")
    assert lines[1].startswith("80,") and '"含,逗號"' in lines[1]  # 原欄位與引號保留
    assert lines[1].split(",")[-3] == f"{r1[0]:g}" and lines[1].split(",")[-2] == pred[0][1]
    assert sum(out["cnt"].values()) == 3
    assert out["mae"] == pytest.approx(sum(abs(a - p) for a, p in zip(amounts, r1)) / 3)


@pytest.mark.req("TC-10", "AC-03.3", "FR-10")
def test_big5_and_utf8_decoding(artifacts):
    text = "撥貸金額,DBR,備註\r\n50,12,客戶王小明\r\n"
    for enc in ("big5", "utf-8"):
        b64 = base64.b64encode(text.encode(enc)).decode()
        assert node(PRED_HTML, "decode", b64) == text


@pytest.mark.req("TC-11", "AC-01.3", "AC-03.1", "FR-05", "FR-10")
def test_warnings_and_daily_filename(artifacts):
    lo, hi = node(PRED_HTML, "meta")["dbr"]
    w = node(PRED_HTML, "warnings", [{"DBR": 30, "無擔保總餘額": -5}, {"DBR": 0}, {"DBR": 12}])
    assert w[0] == [f"DBR 30 超出訓練範圍，已截斷至 {hi:g}", "無擔保總餘額為負值，已視為 0"]
    assert w[1] == [] and w[2] == []  # DBR=0 視為缺值旗標，不警示
    assert node(PRED_HTML, "outName", [2026, 1, 5]) == "預測結果_20260105.csv"


@pytest.mark.req("TC-12", "AC-01.1", "AC-02.1", "UI-01", "UI-02")
@pytest.mark.parametrize("html,expect", [(PRED_HTML, "預測撥貸金額"), (TREE_HTML, "判斷路徑")])
def test_ui_smoke_example_button(artifacts, html, expect):
    """整頁腳本（含畫面程式）在 DOM 替身中可執行；按「範例／重設」產生結果。真瀏覽器另見人工驗收 MT-01。"""
    p = subprocess.run(["node", str(ROOT / "tests/ui_smoke.js"), str(html)], capture_output=True, check=False)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    r = json.loads(p.stdout.decode("utf-8"))
    assert r["inputs"] == 10  # 4 個類別下拉 + 6 個數值欄
    assert expect in r["out"] and "226,578" in r["sub"]


@pytest.mark.req("TC-13", "NFR-05")
@pytest.mark.parametrize("html", [PRED_HTML, TREE_HTML])
def test_no_network_or_external_resources(artifacts, html):
    """客戶資料只在瀏覽器內計算：成品不得含網路呼叫或外部資源。"""
    s = html.read_text(encoding="utf-8")
    for bad in ("fetch(", "XMLHttpRequest", "sendBeacon", "WebSocket", "<script src", "<link ", "<img "):
        assert bad not in s, bad


@pytest.mark.req("TC-14", "NFR-06")
def test_batch_10000_rows_under_3s(artifacts):
    text = "\n".join((ROOT / "regression8.csv").read_text(encoding="utf-8-sig").splitlines()[:10001])
    out = node(PRED_HTML, "batch", text)
    assert out["n"] == 10000 and out["ms"] < 3000, out["ms"]
