# 信貸撥貸金額預測工具

電銷人員在 KYC 後輸入客戶條件，得到參考核准金額；每日名單可以批次預測後，下載結果與主管討論。

| 路徑 | 內容 |
|---|---|
| `docs/02-prd.md`～`docs/06-test-plan.md` | 需求、規格、架構、設計、測試計畫（本專案版） |
| `src/features.py`、`src/train.py` | 特徵規則、訓練、評估、匯出 |
| `templates/*.html` | 兩個工具的畫面範本（模型置換點 `/*__MODEL__*/null`） |
| `dist/*.html` | **交付成品**，雙擊即可離線使用 |
| `Dockerfile`、`deploy/` | Railway 部署設定（nginx 提供 `dist/` 與首頁） |
| `09-啟動說明書.md` | 使用、安裝、訓練、測試與常見問題的完整步驟 |
| `models/`、`reports/` | 模型 JSON、評估報告、測試證據 |
| `tests/` | pytest 加上 Node 測試（直接測 dist 成品） |

```bash
pip install -r requirements.txt     # 另需 Node（只用於測試）
python src/train.py                 # 結束碼 0 = 門檻通過且已更新 dist/；2 = 門檻未過，dist/ 不變
python -m pytest tests -v
```

根目錄的兩個舊 HTML 保留作參考，交付版本以 `dist/` 為準。上線前須先決定 D-01（見 `docs/02-prd.md`）。
