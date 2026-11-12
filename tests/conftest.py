import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "regression8.csv"
PRED_HTML = ROOT / "dist" / "撥貸金額預測工具.html"
TREE_HTML = ROOT / "dist" / "撥貸金額(決策樹判斷).html"
METRICS = ROOT / "reports" / "metrics.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "req(*ids): 追溯到 TC／AC／FR／NFR 編號")


def node(html: Path, op: str, args=None):
    """執行 dist HTML 內的 JS 純函式，回傳 JSON 結果。"""
    req = json.dumps({"html": str(html), "op": op, "args": args}, ensure_ascii=False)
    p = subprocess.run(["node", str(ROOT / "tests" / "js_harness.js")], input=req.encode("utf-8"),
                       capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace"))
    return json.loads(p.stdout.decode("utf-8"))


@pytest.fixture(scope="session")
def artifacts():
    """確認 python src/train.py 已產出受測成品；沒有就失敗，不偷偷重建。"""
    for p in (PRED_HTML, TREE_HTML, METRICS):
        assert p.exists(), f"缺少 {p}，請先執行 python src/train.py"
    return json.loads(METRICS.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def clean_df():
    from features import clean_training
    df, _ = clean_training(pd.read_csv(DATA, encoding="utf-8-sig"))
    return df
