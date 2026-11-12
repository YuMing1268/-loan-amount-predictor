// 以最小 DOM 替身執行整頁 <script>（含畫面程式），按「範例／重設」後輸出結果區 HTML。
// 用法：node tests/ui_smoke.js <html路徑>
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.match(/<script>([\s\S]*)<\/script>/)[1];
const els = {};
const el = id => els[id] || (els[id] = { id, value: "", innerHTML: "", textContent: "", style: {}, onclick: null, onchange: null });
const ctx = { document: { getElementById: el, createElement: () => ({ click() {} }) }, TextDecoder, console };
vm.createContext(ctx);
vm.runInContext(src, ctx);
const ids = (els.form.innerHTML.match(/id="([^"]+)"/g) || []).length;
els.reset.onclick();
process.stdout.write(JSON.stringify({ inputs: ids, sub: els.sub.textContent, out: els.out.innerHTML }));
