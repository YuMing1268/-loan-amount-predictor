// 測試用：載入 dist/*.html 內的 <script>，在 node vm 中執行其純函式。
// 用法：node tests/js_harness.js < request.json
// request: {"html": 路徑, "op": "predict"|"judge"|"batch"|"warnings"|"outName"|"decode"|"meta", "args": ...}
const fs = require("fs"), vm = require("vm");
const req = JSON.parse(fs.readFileSync(0, "utf8"));
const html = fs.readFileSync(req.html, "utf8");
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) throw new Error("找不到 <script>");
const ctx = { module: { exports: {} }, TextDecoder, console };
vm.createContext(ctx);
vm.runInContext(m[1], ctx); // 沒有 document → 不執行畫面程式
const X = ctx.module.exports;
let out;
switch (req.op) {
  case "predict": out = req.args.map(r => { const a = X.predict(r); return [a, X.lvl(a)]; }); break;
  case "judge": out = req.args.map(r => { const j = X.judge(r); return { med: j.leaf.med, n: j.leaf.n, hi: j.hi, steps: j.path.length }; }); break;
  case "batch": { const t0 = Date.now(); const b = X.batchPredict(req.args); const ms = Date.now() - t0; out = b.error ? { error: b.error } : { csv: b.csv, cnt: b.cnt, mae: b.mae, n: b.res.length, ms }; break; }
  case "warnings": out = req.args.map(r => X.warnings(r)); break;
  case "walk": out = req.args.map(f => X.walk(f).path.map(p => p[2])); break;
  case "lvl": out = req.args.map(a => X.lvl(a)); break;
  case "toFeat": out = req.args.map(r => X.toFeat(r)); break;
  case "outName": out = X.outName(new Date(req.args[0], req.args[1] - 1, req.args[2])); break;
  case "decode": out = X.decode(Buffer.from(req.args, "base64")); break;
  case "meta": out = { n: X.M.n, cols: X.M.cols, dbr: X.M.dbr, median: X.M.median, cut: X.M.cut || null }; break;
  default: throw new Error("unknown op " + req.op);
}
process.stdout.write(JSON.stringify(out));
