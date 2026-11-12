# 課堂模板：六份各司其職的工程文件

依 [WORKFLOW](../02-WORKFLOW.md) 進行；不要先複製所有空表。`＜＞` 是替換處，教學提示不放進成品。固定路徑讓通用 skills 及下一個對話能找到成果。

| 模板 → 產出 | 要回答的審查問題 | 何時 |
|---|---|---|
| [01 Brief](01-brief.md) → `docs/01-brief.md` | 問題有何證據，做完怎樣算成功？ | D2 開場 |
| [02 PRD](02-prd.md) → `docs/02-prd.md` | 誰做什麼，成功和拒絕時看見什麼？ | D2 卡 B |
| [03 SRS](03-srs.md) → `docs/03-srs.md` | 規則與邊界是否足以獨立算出預期？ | D2 卡 C |
| [04 SAD](04-sad.md) → `docs/04-sad.md` | 責任、信任邊界、故障與取捨是否合理？ | D2 卡 E |
| [05 SDS](05-sds.md) → `docs/05-sds.md` | 資料與介面契約能否讓兩人獨立實作？ | D2 卡 E |
| [06 Test Plan](06-test-plan.md) → `docs/06-test-plan.md` | 重要風險如何驗、什麼證據支持交付？ | D2 卡 F |

模板以公開工程實務整理，並非 Google 內部模板的複製。版面依讀者要做的事選擇：背景與理由用短段落；有順序或少量例外用列點；要比較同一組欄位、掃描編號或追查證據時用表格。表格儘量維持 2–4 欄，長敘述放在表格外。模板的 `＜＞` 是提示，不要求逐格填表。

只填與本次風險相關的內容；不適用寫原因，未知寫負責人和決策期限，沒有來源的數據記為待量測。每個 Gate 由人審查；狀態為草稿時不可當成已確認輸入。先讓一條需求接到測試與操作結果；[IT 設備借用完整範例](../../02-補充/examples/it-asset/README.md)供課後看深度，其技術棧與規模不代表本課主線。

設計依據：[Google 公開的設計文件與文件維護原則](https://abseil.io/resources/swe-book/html/ch10.html)、[Google code review 的設計／測試檢查](https://google.github.io/eng-practices/review/reviewer/looking-for.html)、[Microsoft 依工作負載取捨品質屬性](https://learn.microsoft.com/en-us/azure/well-architected/what-is-well-architected-framework)。這些是方法來源；本課的六份文件與 Gate 是課程自己的設計。

[TRACEABILITY](07-TRACEABILITY.md)定義編號；[LESSONS](08-LESSONS.md)只記實際偏差與解法，沒有新發現時不湊紀錄。

[agent-rules](../../02-補充/templates/agent-rules/AGENTS.safe.md)是課後其他專案的參考，不覆蓋本 repo 的既有 `AGENTS.md`。本課只在下半段補小組決策。[hooks](../../02-補充/templates/hooks/README.md)是選用示範，完成骨架後再考慮。
