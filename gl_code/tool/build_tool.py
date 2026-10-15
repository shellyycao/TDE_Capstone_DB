"""Build GL_Code_Tool.html from the v2 rules in GL_Code_Apply.ipynb.

Run from anywhere:  python gl_code/tool/build_tool.py
Re-run it whenever the rules in GL_Code_Apply.ipynb change, then send the client the new html.
"""
import datetime
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "GL_Code_Apply.ipynb"
OUT = HERE / "GL_Code_Tool.html"


def load_config():
    nb = json.loads(NOTEBOOK.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {}
    # imports + helpers, ops, rules, tracking inheritance, review settings (same cells the notebook runs)
    for i in [0, 2, 3, 4, 5]:   # imports, operators, rules, tracking inheritance, review settings
        exec(code[i], ns)
    ns["REVIEW_LABEL"] = "review"
    show_cell = [s for s in code if s.startswith("SHOW_COLS")][0]
    exec(show_cell.split("def ledger_path")[0], ns)
    rules = {}
    for comp, lst in ns["RULES_V2"].items():
        rules[comp] = [
            {
                "id": r["id"],
                "priority": r["priority"],
                "gl": r["gl"],
                "why": r.get("why", ""),
                "enabled": r.get("enabled", True),
                "when": [[c, op, v] for c, op, v in r["when"]],
            }
            for r in lst
        ]
    return {
        "rules": rules,
        "own": ns["OWN"],
        "defaultRules": ns["DEFAULT_RULES"],
        "lowConfidenceRules": ns["LOW_CONFIDENCE_RULES"],
        "weakTrackingSources": ns["WEAK_TRACKING_SOURCES"],
        "knownAccounts": ns["KNOWN_ACCOUNTS"],
        "adjustmentTypes": ns["ADJUSTMENT_TYPES"],
        "protectedRules": ns["PROTECTED_RULES"],
        "showCols": ns["SHOW_COLS"],
        "reviewLabel": ns["REVIEW_LABEL"],
        "built": str(datetime.date.today()),
    }


CORE = r"""
// ==CORE-START==
const CFG = __CFG__;

function colIdx(L) { let n = 0; for (const ch of L.toUpperCase()) n = n * 26 + ch.charCodeAt(0) - 64; return n - 1; }
function normalizeGl(code) {
  if (typeof code !== 'string') return code;
  code = code.trim();
  code = code.replace(/[—―]/g, '--');
  code = code.replace(/[‐‑‒–]/g, '-');
  code = code.replace(/^split[\s\-_:]*/i, 'Split-');
  return code;
}
function escRe(s) { return String(s).replace(/[.*+?^${}()|[\]\\\/-]/g, '\\$&'); }
const reCache = {};
function makeRe(vals, word) {
  if (typeof vals === 'string') vals = [vals];
  const alt = vals.map(escRe).join('|');
  const key = (word ? 'w:' : 'c:') + alt;
  if (!reCache[key]) reCache[key] = new RegExp(word ? '(?<![A-Za-z0-9])(?:' + alt + ')(?![A-Za-z0-9])' : '(?:' + alt + ')', 'i');
  return reCache[key];
}

class Frame {
  constructor(rows) { this.rows = rows; this.n = rows.length; this._s = {}; this._l = {}; }
  strip(L) {
    if (!(L in this._s)) { const i = colIdx(L); this._s[L] = this.rows.map(r => (r[i] === undefined ? '' : r[i]).trim()); }
    return this._s[L];
  }
  lower(L) { if (!(L in this._l)) this._l[L] = this.strip(L).map(s => s.toLowerCase()); return this._l[L]; }
}

function condMask(df, cols, op, val) {
  const n = df.n;
  if (op === 'eq_col' || op === 'ne_col') {
    const a = df.lower(cols), b = df.lower(val), out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = op === 'eq_col' ? a[i] === b[i] : a[i] !== b[i];
    return out;
  }
  const masks = [];
  for (const c of cols.split('|')) {
    const m = new Array(n);
    if (op === 'eq') { const v = String(val).trim().toLowerCase(), s = df.lower(c); for (let i = 0; i < n; i++) m[i] = s[i] === v; }
    else if (op === 'in') { const set = new Set(val.map(v => String(v).trim().toLowerCase())), s = df.lower(c); for (let i = 0; i < n; i++) m[i] = set.has(s[i]); }
    else { const re = makeRe(val, op === 'word'), s = df.strip(c); for (let i = 0; i < n; i++) m[i] = re.test(s[i]); }
    masks.push(m);
  }
  const out = new Array(n);
  for (let i = 0; i < n; i++) { let any = false; for (const m of masks) if (m[i]) { any = true; break; } out[i] = any; }
  if (op.startsWith('not_')) for (let i = 0; i < n; i++) out[i] = !out[i];
  return out;
}

function matchRules(df, rules) {
  rules = rules.filter(r => r.enabled !== false);
  const n = df.n, hits = rules.map(r => {
    let m = new Array(n).fill(true);
    for (const [c, op, v] of r.when) { const x = condMask(df, c, op, v); for (let i = 0; i < n; i++) m[i] = m[i] && x[i]; }
    return m;
  });
  const gl = new Array(n).fill(''), rule = new Array(n).fill(''), conflict = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    let best = -1;
    for (let k = 0; k < rules.length; k++) if (hits[k][i] && rules[k].priority > best) best = rules[k].priority;
    if (best < 0) continue;
    let win = -1; const gls = new Set();
    for (let k = 0; k < rules.length; k++) if (hits[k][i] && rules[k].priority === best) { if (win < 0) win = k; gls.add(rules[k].gl); }
    gl[i] = rules[win].gl; rule[i] = rules[win].id; conflict[i] = gls.size > 1;
  }
  return { gl, rule, conflict };
}

function inheritFromTracking(df, gl, rule) {
  gl = gl.slice(); rule = rule.slice();
  const n = df.n, trk = df.strip('N'), aj = df.strip('AJ');
  const adj = aj.map(a => CFG.adjustmentTypes.includes(a));
  const srcRule = new Map(), glSets = new Map(), firstGl = new Map();
  for (let i = 0; i < n; i++) {
    if (adj[i] || trk[i] === '') continue;
    if (!srcRule.has(trk[i])) { srcRule.set(trk[i], rule[i]); firstGl.set(trk[i], gl[i]); glSets.set(trk[i], new Set()); }
    glSets.get(trk[i]).add(gl[i]);
  }
  const fromRule = new Array(n).fill(''), multi = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    if (!adj[i]) continue;
    const t = trk[i], set = glSets.get(t);
    if (set && set.size > 1) multi[i] = true;
    if (set && set.size === 1 && !CFG.protectedRules.includes(rule[i])) {
      const newGl = firstGl.get(t);
      gl[i] = newGl; rule[i] = 'TRACKING'; fromRule[i] = srcRule.get(t) || 'history';
    }
  }
  return { gl, rule, fromRule, multi };
}

function assignGl(df, company) {
  const m = matchRules(df, CFG.rules[company]);
  const inh = inheritFromTracking(df, m.gl, m.rule);
  const n = df.n, gl = inh.gl, rule = inh.rule, fromRule = inh.fromRule, multi = inh.multi;
  const reasons = Array.from({ length: n }, () => []);
  const ownRe = makeRe(CFG.own[company], false), bp = df.strip('BP'), acct = df.strip('C');
  const known = new Set(CFG.knownAccounts[company]);
  const why = {}; for (const r of CFG.rules[company]) why[r.id] = r.why || '';
  why['TRACKING'] = 'same GL as the original shipment (same tracking #)';
  for (let i = 0; i < n; i++) {
    const R = reasons[i], senderOwn = ownRe.test(bp[i]);
    if (!known.has(acct[i])) R.push('new UPS account');
    if (m.conflict[i] && rule[i] !== 'TRACKING') R.push('rules at the same priority disagree');
    if (CFG.defaultRules.includes(rule[i]) && !senderOwn) R.push('only the default rule matched and sender is not the company');
    if (CFG.lowConfidenceRules.includes(rule[i])) R.push('rule was not very accurate on the last answer key');
    if (CFG.weakTrackingSources.includes(fromRule[i])) R.push('original shipment was coded by person/user, corrections often differ');
    if (multi[i] && rule[i] !== 'TRACKING') R.push('tracking # had more than one GL before');
  }
  const need = reasons.map(r => r.length > 0);
  return {
    GL_Code: gl.map((g, i) => need[i] ? CFG.reviewLabel : g), Suggested_GL: gl,
    Status: need.map(x => x ? 'review' : 'auto'), Review_Reason: reasons.map(r => r.join('; ')),
    Rule: rule, Inherited_From: fromRule, Why: rule.map(r => why[r] === undefined ? '' : why[r])
  };
}

// fills column M; rows that already have a value in M are kept
function processRows(rows, company) {
  const df = new Frame(rows), res = assignGl(df, company), M = colIdx('M');
  const n = rows.length;
  for (let i = 0; i < n; i++) {
    const ex = normalizeGl((rows[i][M] === undefined ? '' : rows[i][M]));
    if (ex !== '') { res.GL_Code[i] = ex; res.Status[i] = 'existing'; res.Review_Reason[i] = ''; }
  }
  const out = rows.map((r, i) => { const c = r.slice(); c[M] = res.GL_Code[i]; return c; });
  return { res, out, df };
}

// ---------- csv in / out ----------
function parseCsv(text) {
  const rows = []; let row = [], f = '', q = false, i = 0; const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (q) {
      if (ch === '"') { if (text[i + 1] === '"') { f += '"'; i += 2; continue; } q = false; i++; continue; }
      f += ch; i++; continue;
    }
    if (ch === '"') { q = true; i++; continue; }
    if (ch === ',') { row.push(f); f = ''; i++; continue; }
    if (ch === '\r' || ch === '\n') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(f); f = ''; i++;
      if (!(row.length === 1 && row[0] === '')) rows.push(row);
      row = []; continue;
    }
    f += ch; i++;
  }
  if (f !== '' || row.length) { row.push(f); if (!(row.length === 1 && row[0] === '')) rows.push(row); }
  const w = rows.reduce((a, r) => Math.max(a, r.length), 0);
  for (const r of rows) while (r.length < w) r.push('');
  return rows;
}
function csvField(s) { s = String(s); return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
function toCsv(rows) { return rows.map(r => r.map(csvField).join(',')).join('\n') + '\n'; }

function reviewTable(rows, proc, onlyReview) {
  const { res, df } = proc, n = rows.length, ncols = rows[0] ? rows[0].length : 0;
  const cols = Object.entries(CFG.showCols).filter(([L]) => colIdx(L) < ncols);
  const head = ['Row', 'Final_GL', ...cols.map(([L, name]) => L + ' ' + name), 'GL_Code', 'Suggested_GL', 'Status', 'Review_Reason', 'Rule', 'Inherited_From', 'Why'];
  const body = [];
  for (let i = 0; i < n; i++) {
    if (onlyReview && res.Status[i] !== 'review') continue;
    body.push([i + 1, '', ...cols.map(([L]) => rows[i][colIdx(L)] ?? ''), res.GL_Code[i], res.Suggested_GL[i], res.Status[i], res.Review_Reason[i], res.Rule[i], res.Inherited_From[i], res.Why[i]]);
  }
  return [head, ...body];
}

function detectCompany(name) {
  const u = name.toUpperCase();
  if (u.includes('CSI') || u.includes('COOPER')) return 'CSI';
  if (u.includes('BELIMO')) return 'Belimo';
  return '';
}
// ==CORE-END==
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UPS GL Code Filler</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5d6678;--line:#dde1e8;--accent:#1b2a5c;--accent-ink:#fff;--ok:#12805c;--warn:#b25e00;--warn-bg:#fff4e0;--ok-bg:#e6f6ef}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e8ebf2;--muted:#9aa4b5;--line:#2c3340;--accent:#3a54a8;--accent-ink:#fff;--ok:#43c79a;--warn:#ffb454;--warn-bg:#3a2c12;--ok-bg:#12332a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:900px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:16px}
.step{display:flex;gap:12px;align-items:center;margin:0 0 12px}
.step b{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--accent);color:var(--accent-ink);font-size:14px}
.step h2{font-size:18px;margin:0}
.drop{border:2px dashed var(--line);border-radius:10px;padding:28px;text-align:center;color:var(--muted);cursor:pointer}
.drop.over{border-color:var(--accent);color:var(--ink)}
button,.btn{background:var(--accent);color:var(--accent-ink);border:0;border-radius:8px;padding:10px 16px;font:inherit;font-weight:600;cursor:pointer}
button.secondary,.btn.secondary{background:transparent;color:var(--accent);border:1px solid var(--accent)}
button:disabled{opacity:.45;cursor:not-allowed}
select{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.file{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}
.file:first-child{border-top:0}
.file .name{flex:1 1 260px;word-break:break-all}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:13px;font-weight:600}
.pill.ok{background:var(--ok-bg);color:var(--ok)} .pill.warn{background:var(--warn-bg);color:var(--warn)}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.stat{flex:1 1 130px;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat .n{font-size:24px;font-weight:700} .stat .l{color:var(--muted);font-size:13px}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600} .scroll{overflow-x:auto}
.fileblock{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}.fileblock:first-of-type{border-top:0;padding-top:0;margin-top:0}.fileblock h3{font-size:16px;margin:0 0 4px;word-break:break-all}
.note{color:var(--muted);font-size:14px} .err{color:var(--warn);font-weight:600}
</style>
</head>
<body>
<main>
<h1>UPS GL Code Filler</h1>
<p class="sub">Add your UPS billing file. Get the same file back with column M (GL code) filled in. Your file stays on this computer; nothing is uploaded.</p>

<section class="card">
  <div class="step"><b>1</b><h2>Add your UPS file</h2></div>
  <div class="drop" id="drop">Drag the UPS .csv file here, or click to choose<br><span class="note">Use the file exactly as exported from UPS (no header row). Column M can be empty.</span></div>
  <input type="file" id="pick" accept=".csv" multiple hidden>
  <div id="files"></div>
  <div class="actions"><button id="run" disabled>Fill GL codes</button></div>
  <p class="note" id="msg"></p>
</section>

<div id="results"><section class="card"><div class="step"><b>2</b><h2>Download your results</h2></div><p class="note">Your results appear here after you click <b>Fill GL codes</b>.</p></section></div>

<section class="card">
  <div class="step"><b>3</b><h2>Optional: put your reviewed answers back in</h2></div>
  <p class="note">Open the review sheet in Excel, type the right code in the <b>Final_GL</b> column for the rows you checked, and save it as .csv. Then add both files here: the filled file (ending in <code>_GL.csv</code>) and your saved review sheet.</p>
  <label>Filled file (<code>_GL.csv</code>): <input type="file" id="glfile" accept=".csv"></label><br><br>
  <label>Your reviewed sheet: <input type="file" id="revfile" accept=".csv"></label>
  <div class="actions"><button id="apply" class="secondary" disabled>Apply my answers</button></div>
  <p class="note" id="applymsg"></p>
</section>
<p class="note">Rules version built __BUILT__ from GL_Code_Apply.ipynb (v2). Rows marked <b>review</b> are ones the rules are not sure about; a person should check them.</p>
</main>

<script>
__CORE__

// ---------- page ----------
const $ = id => document.getElementById(id);
const state = { files: [], results: [] };
const PLACEHOLDER = $('results').innerHTML;

function decode(buf) {
  try { return { text: new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(buf), enc: 'utf-8' }; }
  catch (e) { return { text: new TextDecoder('windows-1252').decode(buf), enc: 'latin1' }; }
}
function encode(text, enc) {
  if (enc === 'utf-8') return new TextEncoder().encode(text);
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) { const c = text.charCodeAt(i); out[i] = c < 256 ? c : 63; }
  return out;
}
function download(name, bytes, mime) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([bytes], { type: mime || 'text/csv' }));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function addFiles(list) {
  for (const f of list) if (/\.csv$/i.test(f.name)) state.files.push({ file: f, company: detectCompany(f.name) });
  renderFiles();
}
function renderFiles() {
  $('files').innerHTML = state.files.map((x, i) => {
    const isKey = /completed/i.test(x.file.name) && !/not[ _]completed/i.test(x.file.name);
    return `<div class="file"><span class="name">${esc(x.file.name)}</span>
      <label>Company <select data-i="${i}"><option value="">choose...</option><option value="CSI"${x.company === 'CSI' ? ' selected' : ''}>CooperSurgical (CSI)</option><option value="Belimo"${x.company === 'Belimo' ? ' selected' : ''}>Belimo</option></select></label>
      ${isKey ? '<span class="pill warn">looks like an already-completed file</span>' : ''}
      <button class="secondary" data-rm="${i}">Remove</button></div>`;
  }).join('');
  $('files').querySelectorAll('select').forEach(s => s.onchange = () => { state.files[+s.dataset.i].company = s.value; check(); });
  $('files').querySelectorAll('[data-rm]').forEach(b => b.onclick = () => { state.files.splice(+b.dataset.rm, 1); renderFiles(); });
  check();
}
function check() { $('run').disabled = !state.files.length || state.files.some(x => !x.company); }

$('drop').onclick = () => $('pick').click();
$('pick').onchange = e => { addFiles(e.target.files); e.target.value = ''; };
['dragover', 'dragenter'].forEach(ev => $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.remove('over'); }));
$('drop').addEventListener('drop', e => addFiles(e.dataTransfer.files));

$('run').onclick = async () => {
  $('msg').textContent = 'Working...'; $('msg').className = 'note'; $('results').innerHTML = ''; state.results = [];
  await new Promise(r => setTimeout(r, 30));
  try {
    for (const x of state.files) {
      const { text, enc } = decode(await x.file.arrayBuffer());
      const rows = parseCsv(text);
      if (!rows.length) throw new Error(x.file.name + ' is empty');
      if (rows[0].length < 82) throw new Error(x.file.name + ' has only ' + rows[0].length + ' columns. A UPS billing file has about 244. Is this the right file, and does it have no header row?');
      const proc = processRows(rows, x.company);
      state.results.push({ name: x.file.name, company: x.company, rows, enc, proc });
    }
    render(); $('msg').textContent = '';
  } catch (e) { $('results').innerHTML = PLACEHOLDER; $('msg').textContent = e.message; $('msg').className = 'err'; }
};

function baseName(n) { return n.replace(/\.csv$/i, '').replace(/[ _]Not[ _]Completed/i, ''); }

function render() {
  $('results').innerHTML = '<section class="card"><div class="step"><b>2</b><h2>Download your results</h2></div>' + state.results.map((r, k) => {
    const c = { auto: 0, review: 0, existing: 0 }; r.proc.res.Status.forEach(s => c[s]++);
    const rev = reviewTable(r.rows, r.proc, true), shown = rev.slice(1, 11);
    const headIdx = ['Row', 'GL_Code', 'Suggested_GL', 'Review_Reason'].map(h => rev[0].indexOf(h));
    return `<div class="fileblock"><h3>${esc(r.name)}</h3>
      <div class="stats">
        <div class="stat"><div class="n">${r.rows.length.toLocaleString()}</div><div class="l">rows</div></div>
        <div class="stat"><div class="n" style="color:var(--ok)">${c.auto.toLocaleString()}</div><div class="l">filled automatically</div></div>
        <div class="stat"><div class="n" style="color:var(--warn)">${c.review.toLocaleString()}</div><div class="l">need a person to check (marked "review")</div></div>
        <div class="stat"><div class="n">${c.existing.toLocaleString()}</div><div class="l">already had a code (kept)</div></div>
      </div>
      <div class="actions">
        <button data-dl="gl" data-k="${k}">Download filled file</button>
        <button class="secondary" data-dl="rev" data-k="${k}">Download review sheet (${c.review} rows)</button>
        <button class="secondary" data-dl="all" data-k="${k}">Download all rows with reasons</button>
      </div>
      ${c.review ? `<p class="note">First ${shown.length} of ${c.review} rows to check:</p><div class="scroll"><table><tr><th>Row</th><th>Suggested GL</th><th>Why it needs a check</th></tr>
        ${shown.map(s => `<tr><td>${s[headIdx[0]]}</td><td>${esc(s[headIdx[2]])}</td><td>${esc(s[headIdx[3]])}</td></tr>`).join('')}</table></div>` : '<p class="note">Nothing needs checking.</p>'}
    </div>`;
  }).join('') + '</section>';
  $('results').querySelectorAll('[data-dl]').forEach(b => b.onclick = () => {
    const r = state.results[+b.dataset.k], base = baseName(r.name);
    if (b.dataset.dl === 'gl') download(base + '_GL.csv', encode(toCsv(r.proc.out), r.enc));
    else { const t = reviewTable(r.rows, r.proc, b.dataset.dl === 'rev'); download(base + (b.dataset.dl === 'rev' ? '_GL_review.csv' : '_GL_all_rows.csv'), encode('﻿' + toCsv(t), 'utf-8')); }
  });
}

// ---------- step 3 ----------
const apply = { gl: null, rev: null };
async function pick(id, key) { $(id).onchange = async e => { apply[key] = e.target.files[0] || null; $('apply').disabled = !(apply.gl && apply.rev); }; }
pick('glfile', 'gl'); pick('revfile', 'rev');
$('apply').onclick = async () => {
  $('applymsg').className = 'note'; $('applymsg').textContent = '';
  try {
    const g = decode(await apply.gl.arrayBuffer()), rows = parseCsv(g.text);
    const rv = parseCsv(decode(await apply.rev.arrayBuffer()).text.replace(/^﻿/, ''));
    const head = rv[0].map(h => h.trim()), iRow = head.indexOf('Row'), iFin = head.indexOf('Final_GL');
    if (iRow < 0 || iFin < 0) throw new Error('The review sheet needs the "Row" and "Final_GL" columns. Use the sheet this tool gave you.');
    const M = colIdx('M'); let n = 0;
    for (const r of rv.slice(1)) {
      const fin = normalizeGl(r[iFin] || ''), row = parseInt(r[iRow], 10);
      if (fin && row >= 1 && row <= rows.length) { rows[row - 1][M] = fin; n++; }
    }
    const still = rows.filter(r => (r[M] || '') === CFG.reviewLabel).length;
    download(apply.gl.name.replace(/\.csv$/i, '') + '_final.csv', encode(toCsv(rows), g.enc));
    $('applymsg').textContent = `Updated ${n} rows. ${still} rows are still marked "review".`;
  } catch (e) { $('applymsg').className = 'err'; $('applymsg').textContent = e.message; }
};
</script>
</body>
</html>
"""


def main():
    cfg = load_config()
    core = CORE.replace("__CFG__", json.dumps(cfg, ensure_ascii=False))
    OUT.write_text(HTML.replace("__CORE__", core).replace("__BUILT__", cfg["built"]), encoding="utf-8")
    n = {k: len(v) for k, v in cfg["rules"].items()}
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB) with rules {n}")


if __name__ == "__main__":
    main()
