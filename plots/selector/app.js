/* Selettore run: filtri -> /api/query (copertura) e /api/preview (grafico). */

const state = {
  // colonna -> {op, values}: l'operatore decide se i valori scelti si tengono
  // (è / fra) o si escludono (non è / non fra), e se se ne puo' scegliere piu' di uno
  dims: {},
  seeds: { min: null, max: null },
  // run tolte a mano dalla tabella di copertura: restano visibili in tabella
  // ma non entrano nel grafico
  excluded: new Set(),
  // i colori delle serie sono sempre automatici (una curva per ogni
  // configurazione che varia): niente controllo in pagina per sceglierli a mano.
  // kind: "curve" (apprendimento, nel tempo) | "budget" (eval finale per livello
  // di budget): cambiano sia i dati (curves.py vs budget.py) sia i controlli
  // disponibili (smooth non ha senso per il budget).
  // budget_x non e' scelto dalla pagina: resta il default del backend
  // (budget_level, robusto per ogni arm — vedi rtplots/figure.py).
  grid: { kind: "curve", rows: "", cols: "", band: "se", smooth: 5,
          metric: "", legend: "outside_right", panel_size: [3.4, 2.5] },
  // ritocchi a mano delle serie: etichetta di partenza -> {name, color}
  series_overrides: {},
  series: [],            // ultimo elenco di serie disegnate
  seriesSel: null,       // quale e' selezionata nell'elenco
  palette: [],
  meta: null,
};

const $ = (sel) => document.querySelector(sel);
let queryTimer = null;
let selectionItems = [];   // ultimo elenco di selezioni salvate, per ridisegnarlo

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

/* --- filtri --------------------------------------------------------------- */

const pillIndex = {};   // colonna -> valore -> {input, label, countEl}
const opIndex = {};     // colonna -> <select> dell'operatore

function asFilter(raw) {
  if (Array.isArray(raw)) return { op: state.meta?.default_op || "in", values: [...raw] };
  return { op: raw?.op || state.meta?.default_op || "in", values: [...(raw?.values || [])] };
}

function filterOf(col) {
  if (!state.dims[col]) state.dims[col] = asFilter(null);
  return state.dims[col];
}

function isMulti(op) {
  const found = (state.meta?.ops || []).find((o) => o.op === op);
  return found ? found.multi : true;
}

function isNegative(op) {
  return op === "is_not" || op === "not_in";
}

function renderDimensions(dims) {
  const box = $("#dimensions");
  box.innerHTML = "";
  const ops = state.meta?.ops || [];
  for (const dim of dims) {
    const sec = document.createElement("section");
    sec.className = "dim";
    const head = document.createElement("div");
    head.className = "dim-head";
    head.innerHTML = `<h2>${esc(dim.title)}</h2>` +
      `<select class="op" aria-label="operatore per ${esc(dim.title)}">` +
      ops.map((o) => `<option value="${esc(o.op)}">${esc(o.label)}</option>`).join("") +
      `</select>`;
    const opSel = head.querySelector("select");
    opSel.value = filterOf(dim.col).op;
    opIndex[dim.col] = opSel;
    sec.appendChild(head);

    const pills = document.createElement("div");
    pills.className = "pills";
    pillIndex[dim.col] = {};
    for (const v of dim.values) {
      const label = document.createElement("label");
      label.className = "pill";
      label.innerHTML = `<input type="checkbox" value="${esc(v.value)}">` +
        `<span>${v.label}</span><span class="count">${v.count}</span>`;
      const input = label.querySelector("input");
      input.addEventListener("change", () => {
        const f = filterOf(dim.col);
        const cur = new Set(f.values);
        input.checked ? cur.add(v.value) : cur.delete(v.value);
        f.values = (!isMulti(f.op) && input.checked) ? [v.value] : [...cur];
        syncPills(dim.col);
        state.excluded.clear();
        scheduleQuery();
      });
      pills.appendChild(label);
      pillIndex[dim.col][v.value] = { input, label, countEl: label.querySelector(".count") };
    }
    sec.appendChild(pills);
    syncPills(dim.col);

    opSel.addEventListener("change", () => {
      const f = filterOf(dim.col);
      f.op = opSel.value;
      if (!isMulti(f.op) && f.values.length > 1) f.values = [f.values[0]];
      syncPills(dim.col);
      state.excluded.clear();
      scheduleQuery();
    });
    box.appendChild(sec);
  }
}

function syncPills(col) {
  const f = filterOf(col);
  const chosen = new Set(f.values.map(String));
  const negative = isNegative(f.op);
  for (const [value, ref] of Object.entries(pillIndex[col] || {})) {
    ref.input.checked = chosen.has(String(value));
    ref.label.classList.toggle("on", ref.input.checked && !negative);
    ref.label.classList.toggle("off", ref.input.checked && negative);
  }
  if (opIndex[col]) {
    opIndex[col].value = f.op;
    opIndex[col].classList.toggle("negative", negative);
  }
}

function applyCounts(counts) {
  if (!counts) return;
  for (const [col, values] of Object.entries(pillIndex)) {
    const colCounts = counts[col] || {};
    for (const [value, ref] of Object.entries(values)) {
      const n = colCounts[value] || 0;
      ref.countEl.textContent = n;
      ref.label.classList.toggle("zero", n === 0);
    }
  }
}

/* Tendina "cosa plottare": dipende dal tipo di grafico (curva/budget). */
function renderMetrics() {
  const groups = state.grid.kind === "budget" ? state.meta.metrics_budget : state.meta.metrics_curve;
  const dflt = state.grid.kind === "budget"
    ? state.meta.default_metric_budget : state.meta.default_metric_curve;
  const sel = $("#grid-metric");
  sel.innerHTML = groups.map((g) =>
    `<optgroup label="${esc(g.group)}">` +
    g.options.map((o) => `<option value="${esc(o.key)}">${esc(o.label)}</option>`).join("") +
    `</optgroup>`).join("");
  if (!state.grid.metric || !groups.some((g) => g.options.some((o) => o.key === state.grid.metric))) {
    state.grid.metric = dflt;
  }
  sel.value = state.grid.metric;
}

function syncKindVisibility() {
  const isBudget = state.grid.kind === "budget";
  $("#grid-smooth-field").hidden = isBudget;
  const iqrOpt = $("#grid-band").querySelector('option[value="iqr"]');
  if (iqrOpt) iqrOpt.hidden = false; // iqr supportato in entrambi i casi
}

function renderKindControls() {
  const box = $("#grid-kind");
  for (const input of box.querySelectorAll('input[name="kind"]')) {
    input.checked = input.value === state.grid.kind;
    input.closest(".pill").classList.toggle("on", input.checked);
    input.addEventListener("change", () => {
      if (!input.checked) return;
      state.grid.kind = input.value;
      for (const el of box.querySelectorAll(".pill")) {
        el.classList.toggle("on", el.querySelector("input").checked);
      }
      syncKindVisibility();
      renderMetrics();
      maybeAutoPreview();
    });
  }
  syncKindVisibility();
}

function renderGridControls(fields) {
  for (const [id, key] of [["#grid-rows", "rows"], ["#grid-cols", "cols"]]) {
    const sel = $(id);
    sel.innerHTML = `<option value="">—</option>` +
      fields.map((f) => `<option value="${f.col}">${f.title}</option>`).join("");
    sel.value = state.grid[key];
    sel.addEventListener("change", () => {
      state.grid[key] = sel.value;
      maybeAutoPreview();
    });
  }
  $("#grid-metric").addEventListener("change", (e) => {
    state.grid.metric = e.target.value; maybeAutoPreview();
  });
  $("#grid-band").addEventListener("change", (e) => {
    state.grid.band = e.target.value; maybeAutoPreview();
  });
  $("#grid-legend").addEventListener("change", (e) => {
    state.grid.legend = e.target.value; maybeAutoPreview();
  });
  for (const [id, i] of [["#grid-panel-w", 0], ["#grid-panel-h", 1]]) {
    $(id).addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      if (Number.isFinite(v) && v > 0) state.grid.panel_size[i] = v;
      maybeAutoPreview();
    });
  }
  $("#grid-smooth").addEventListener("change", (e) => {
    state.grid.smooth = parseInt(e.target.value, 10) || 1; maybeAutoPreview();
  });
}

/* --- query ---------------------------------------------------------------- */

function payload() {
  return { dims: state.dims, seeds: state.seeds, grid: state.grid,
           excluded: [...state.excluded],
           series_overrides: state.series_overrides };
}

function scheduleQuery() {
  clearTimeout(queryTimer);
  queryTimer = setTimeout(runQuery, 180);
}

async function runQuery() {
  const data = await post("/api/query", payload());
  if (data.error) { toast(data.error); return; }
  $("#n-runs").textContent = data.n_runs;
  $("#n-configs").textContent = data.n_configs;
  $("#states").innerHTML = Object.entries(data.states || {})
    .map(([k, v]) => `<span class="badge ${k}">${k}: ${v}</span>`).join("") +
    (data.n_excluded ? `<span class="badge excluded">escluse: ${data.n_excluded}</span>` : "");
  state.filterArgs = data.filter_args || [];
  applyCounts(data.counts);
  renderCoverage(data.coverage);
  maybeAutoPreview();
}

function renderCoverage(cov) {
  const box = $("#coverage");
  if (!cov || !cov.rows.length) {
    box.innerHTML = `<p class="placeholder">Nessuna run per questi filtri.</p>`;
    $("#coverage-note").textContent = "";
    return;
  }
  const maxSeeds = Math.max(...cov.rows.map((r) => r.n_seeds));
  const allOn = cov.rows.every((r) => r.on);
  const head =
    `<th class="pick"><input type="checkbox" id="cov-all"${allOn ? " checked" : ""}` +
    ` title="tutte / nessuna"></th>` +
    cov.columns.map((c) => `<th>${c}</th>`).join("") +
    `<th>run</th><th>seed</th><th>quali seed</th>`;
  const body = cov.rows.map((r, i) => {
    const partial = r.n_seeds < maxSeeds;
    const cls = [partial ? "partial" : "", r.on ? "" : "off"].filter(Boolean).join(" ");
    return `<tr class="${cls}">` +
      `<td class="pick"><input type="checkbox" data-row="${i}"${r.on ? " checked" : ""}></td>` +
      r.cells.map((c) => `<td>${c}</td>`).join("") +
      `<td class="num">${r.n_runs}</td>` +
      `<td class="num seeds-missing">${r.n_seeds}</td>` +
      `<td class="num">${r.seeds}</td></tr>`;
  }).join("");
  box.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  const setRow = (row, on) => {
    for (const id of row.run_ids) on ? state.excluded.delete(id) : state.excluded.add(id);
  };
  for (const input of box.querySelectorAll("input[data-row]")) {
    input.addEventListener("change", () => {
      setRow(cov.rows[Number(input.dataset.row)], input.checked);
      runQuery();
    });
  }
  $("#cov-all").addEventListener("change", (e) => {
    for (const row of cov.rows) setRow(row, e.target.checked);
    runQuery();
  });

  const note = $("#coverage-note");
  const off = cov.rows.filter((r) => !r.on).length;
  const notes = [];
  if (cov.truncated) notes.push("(elenco troncato)");
  if (off) notes.push(`${off} combinazioni escluse dal grafico`);
  note.textContent = notes.join(" · ");
  if (off) {
    const btn = document.createElement("button");
    btn.className = "ghost small";
    btn.textContent = "includi tutte";
    btn.addEventListener("click", () => { state.excluded.clear(); runQuery(); });
    note.append(" ", btn);
  }
}

/* --- anteprima ------------------------------------------------------------ */

function maybeAutoPreview() {
  if ($("#auto-preview").checked) runPreview();
}

async function runPreview() {
  const box = $("#preview-box");
  box.innerHTML = `<p class="spinner">Disegno in corso…</p>`;
  const data = await post("/api/preview", payload());
  if (data.error) { box.innerHTML = `<p class="error">${data.error}</p>`; return; }
  const hue = (data.hue || []).join(" × ");
  const merged = (data.merged || []).join(", ");
  box.innerHTML = `<img src="data:image/png;base64,${data.png}" alt="anteprima">` +
    `<p class="hint">${esc(data.metric || "")} · ${data.series} serie · ` +
    `${data.panels} pannelli · ${data.elapsed}s` +
    (hue ? ` · colori${data.auto_hue ? " (auto)" : ""}: ${esc(hue)}` : "") + `</p>` +
    (merged ? `<p class="error merged-warning">${esc(merged)} ${
      data.merged.length > 1 ? "variano" : "varia"} senza separare le curve: ` +
      `configurazioni diverse sono mediate insieme.</p>` : "");
  renderSeries(data.series_list || [], data.palette || []);
}

/* --- ritocchi alle serie -------------------------------------------------- */

function renderSeries(items, palette) {
  state.series = items;
  state.palette = palette;
  const box = $("#series-box");
  box.hidden = items.length === 0;
  const list = $("#series-list");
  list.innerHTML = "";
  for (const s of items) {
    const el = document.createElement("button");
    el.className = "series-item" + (state.seriesSel === s.key ? " on" : "");
    el.innerHTML = `<span class="swatch" style="background:${esc(s.color)}"></span>` +
      `<span class="series-name">${esc(s.label)}</span>` +
      (s.renamed || s.recolored ? `<span class="badge tweak">ritoccata</span>` : "");
    el.addEventListener("click", () => {
      state.seriesSel = state.seriesSel === s.key ? null : s.key;
      renderSeries(items, palette);
    });
    list.appendChild(el);
  }
  renderSeriesEdit();
}

function renderSeriesEdit() {
  const chosen = (state.series || []).find((s) => s.key === state.seriesSel);
  $("#series-edit").hidden = !chosen;
  if (!chosen) return;
  $("#series-name").value = chosen.label;
  const box = $("#series-palette");
  box.innerHTML = "";
  for (const color of state.palette || []) {
    const el = document.createElement("button");
    el.className = "swatch pick" + (color === chosen.color ? " on" : "");
    el.style.background = color;
    el.title = color;
    el.addEventListener("click", () => tweakSeries(chosen.key, { color }));
    box.appendChild(el);
  }
}

function tweakSeries(key, change) {
  const cur = state.series_overrides[key] || {};
  state.series_overrides[key] = { ...cur, ...change };
  runPreview();
}

function resetSeries() {
  if (!state.seriesSel) return;
  delete state.series_overrides[state.seriesSel];
  runPreview();
}

function copySeriesRule() {
  const chosen = (state.series || []).find((s) => s.key === state.seriesSel);
  if (!chosen) return;
  navigator.clipboard.writeText(chosen.rule)
    .then(() => toast("Regola copiata: incollala in plots/style.toml"))
    .catch(() => toast("Copia non riuscita"));
}

/* --- export --------------------------------------------------------------- */

function download(url, filename) {
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.rel = "noopener";
  document.body.appendChild(a); a.click(); a.remove();
}

function showDownloadLink(data) {
  const box = $("#download-box");
  box.hidden = false;
  box.innerHTML = `Se il download non parte: ` +
    `<a href="${data.url}" download="${esc(data.filename)}">${esc(data.filename)}</a>` +
    ` · copia sul server: <code>${esc(data.saved_path)}</code>`;
}

async function exportFigure(format) {
  const btn = format === "tex" ? $("#export-latex") : $("#export-jpeg");
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = "…";
  try {
    const data = await post("/api/export", { ...payload(), format });
    if (data.error) { toast(data.error); return; }
    download(data.url, data.filename);
    showDownloadLink(data);
    toast(format === "tex"
      ? `${data.n_panels} pannelli + snippet · ${data.filename}`
      : data.filename);
  } finally {
    btn.disabled = false; btn.textContent = old;
  }
}

/* --- azioni --------------------------------------------------------------- */

async function save() {
  const name = $("#save-name").value.trim();
  const data = await post("/api/save", { ...payload(), name });
  if (data.error) { toast(data.error); return; }
  renderSelections(data.items);
  toast(`«${data.name}» salvata: ${data.n_runs} run`);
}

/* --- selezioni salvate ----------------------------------------------------- */

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderSelections(items) {
  selectionItems = items || [];
  const box = $("#selections");
  if (!selectionItems.length) {
    box.innerHTML = `<p class="placeholder">Nessuna selezione salvata.</p>`;
    return;
  }
  box.innerHTML = selectionItems.map((it) => `
    <div class="sel-item" data-slug="${esc(it.slug)}">
      <div class="sel-load" role="button" tabindex="0">
        <strong>${esc(it.name)}</strong>
        <span class="hint filters">${esc(it.summary)}</span>
      </div>
      <button class="sel-rename ghost small" title="rinomina">✎</button>
      <button class="sel-del ghost small" title="elimina">✕</button>
    </div>`).join("");
  box.querySelectorAll(".sel-item").forEach((item) => {
    const slug = item.dataset.slug;
    const load = item.querySelector(".sel-load");
    load.addEventListener("click", () => loadSelection(slug));
    load.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); loadSelection(slug); }
    });
    item.querySelector(".sel-rename")
      .addEventListener("click", () => startRename(item, slug));
    item.querySelector(".sel-del")
      .addEventListener("click", () => deleteSelection(slug));
  });
}

function startRename(item, slug) {
  const nameEl = item.querySelector("strong");
  if (!nameEl) return;
  const old = nameEl.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "sel-name-edit";
  input.value = old;
  input.maxLength = 60;
  input.addEventListener("click", (e) => e.stopPropagation());
  nameEl.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const finish = async (commit) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (!commit || !name || name === old) {
      renderSelections(selectionItems);
      return;
    }
    const data = await post("/api/selections/rename", { slug, name });
    if (data.error) { toast(data.error); renderSelections(selectionItems); return; }
    renderSelections(data.items);
    toast(`Rinominata: «${data.name}»`);
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

function setPanel(open) {
  $("#sel-card").classList.toggle("closed", !open);
  $("#sel-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  try { localStorage.setItem("selpanel", open ? "1" : "0"); } catch (e) { /* ignora */ }
}

async function loadSelection(slug) {
  const data = await post("/api/selections/load", { slug });
  if (data.error) { toast(data.error); return; }
  applySelection(data);
  toast(`«${data.name}» applicata: ${data.n_runs} run`);
}

async function deleteSelection(slug) {
  const data = await post("/api/selections/delete", { slug });
  renderSelections(data.items);
  toast("Selezione eliminata");
}

function gridFromSpec(spec) {
  const out = {};
  for (const k of ["kind", "rows", "cols", "band", "smooth", "metric", "budget_x", "legend"]) {
    if (spec[k] !== undefined && spec[k] !== null && spec[k] !== "") out[k] = spec[k];
  }
  if (Array.isArray(spec.panel_size) && spec.panel_size.length === 2) out.panel_size = spec.panel_size;
  return out;
}

function applySelection(entry) {
  state.dims = {};
  for (const [col, raw] of Object.entries(entry.dims || {})) state.dims[col] = asFilter(raw);
  state.seeds = entry.seeds || { min: null, max: null };
  state.excluded = new Set(entry.excluded || []);
  state.series_overrides = entry.series_overrides || (entry.spec || {}).series_overrides || {};
  state.seriesSel = null;
  state.grid = { ...state.grid, ...gridFromSpec(entry.spec || {}) };
  $("#save-name").value = entry.name || "";

  for (const col of Object.keys(pillIndex)) syncPills(col);
  $("#seed-min").value = state.seeds.min ?? state.meta.seed_min;
  $("#seed-max").value = state.seeds.max ?? state.meta.seed_max;
  for (const input of $("#grid-kind").querySelectorAll('input[name="kind"]')) {
    input.checked = input.value === state.grid.kind;
    input.closest(".pill").classList.toggle("on", input.checked);
  }
  syncKindVisibility();
  renderMetrics();
  $("#grid-metric").value = state.grid.metric;
  $("#grid-rows").value = state.grid.rows || "";
  $("#grid-cols").value = state.grid.cols || "";
  $("#grid-band").value = state.grid.band || "se";
  $("#grid-smooth").value = state.grid.smooth || 5;
  $("#grid-legend").value = state.grid.legend || "outside_right";
  $("#grid-panel-w").value = state.grid.panel_size?.[0] ?? 3.4;
  $("#grid-panel-h").value = state.grid.panel_size?.[1] ?? 2.5;
  runQuery();
}

function copyFilters() {
  const text = (state.filterArgs || []).join(" ");
  navigator.clipboard.writeText(text)
    .then(() => toast(text ? `Copiato: ${text}` : "Nessun filtro attivo"))
    .catch(() => toast(text || "nessun filtro"));
}

function reset() {
  state.dims = {};
  state.seeds = { min: null, max: null };
  state.excluded.clear();
  state.series_overrides = {};
  state.seriesSel = null;
  for (const col of Object.keys(pillIndex)) syncPills(col);
  $("#seed-min").value = state.meta.seed_min;
  $("#seed-max").value = state.meta.seed_max;
  runQuery();
}

/* --- avvio ---------------------------------------------------------------- */

async function init() {
  const meta = await (await fetch("/api/dimensions")).json();
  state.meta = meta;
  state.grid.metric = meta.default_metric_curve;
  renderDimensions(meta.dimensions);
  renderKindControls();
  renderMetrics();
  renderGridControls(meta.grid_fields);
  renderSelections(meta.selections);
  $("#selection-path").textContent = meta.selection_path;
  $("#save-name").addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
  for (const [id, key] of [["#seed-min", "min"], ["#seed-max", "max"]]) {
    const el = $(id);
    el.value = key === "min" ? meta.seed_min : meta.seed_max;
    el.min = meta.seed_min; el.max = meta.seed_max;
    el.addEventListener("change", () => {
      const v = parseInt(el.value, 10);
      state.seeds[key] = Number.isFinite(v) ? v : null;
      scheduleQuery();
    });
  }
  $("#preview").addEventListener("click", runPreview);
  $("#series-reset").addEventListener("click", resetSeries);
  $("#series-rule").addEventListener("click", copySeriesRule);
  $("#series-name").addEventListener("change", (e) => {
    if (state.seriesSel) tweakSeries(state.seriesSel, { name: e.target.value });
  });
  $("#export-jpeg").addEventListener("click", () => exportFigure("jpeg"));
  $("#export-latex").addEventListener("click", () => exportFigure("tex"));
  $("#sel-toggle").addEventListener("click", () =>
    setPanel($("#sel-card").classList.contains("closed")));
  let open = true;
  try { open = localStorage.getItem("selpanel") !== "0"; } catch (e) { /* ignora */ }
  setPanel(open);
  $("#save").addEventListener("click", save);
  $("#copy").addEventListener("click", copyFilters);
  $("#reset").addEventListener("click", reset);
  runQuery();
}

init();
