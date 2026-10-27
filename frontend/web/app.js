/* SciVerify web app. Plain JavaScript, no build step, no network access beyond this server. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
  const fmtTime = (s) => (s < 10 ? s.toFixed(1) : Math.round(s)) + " s";

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage full or blocked */ }
    },
  };

  async function api(path, options = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    if (!res.ok) throw new Error(await errorText(res));
    return res.json();
  }

  async function errorText(res) {
    try {
      const body = await res.json();
      if (typeof body.detail === "string") return body.detail;
      if (Array.isArray(body.detail) && body.detail[0]) return body.detail[0].msg;
    } catch (e) { /* not JSON */ }
    return `${res.status} ${res.statusText}`;
  }

  let toastTimer;
  function toast(msg) {
    const el = $("#toast");
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 2400);
  }

  function download(filename, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = Object.assign(document.createElement("a"), { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function copy(text, done) {
    try { await navigator.clipboard.writeText(text); toast(done); }
    catch (e) { window.prompt("Copy this:", text); }
  }

  /* Wrap claim words inside `text` in <mark>, escaping everything else. */
  function markTerms(text, terms) {
    const set = new Set((terms || []).map((t) => t.toLowerCase()));
    if (!set.size) return esc(text);
    let out = "", last = 0;
    for (const m of text.matchAll(/[A-Za-z0-9]+/g)) {
      if (!set.has(m[0].toLowerCase())) continue;
      out += esc(text.slice(last, m.index)) + "<mark>" + esc(m[0]) + "</mark>";
      last = m.index + m[0].length;
    }
    return out + esc(text.slice(last));
  }

  const LABELS = {
    SUPPORTED: "Supported", CONTRADICTED: "Contradicted", CONFLICTING: "Conflicting", UNKNOWN: "Unknown",
    SUPPORT: "Supports", CONTRADICT: "Contradicts", NEUTRAL: "Neutral",
  };
  const HEADLINES = {
    SUPPORTED: "The retrieved papers support this claim.",
    CONTRADICTED: "The retrieved papers contradict this claim.",
    CONFLICTING: "The retrieved papers disagree with each other.",
    UNKNOWN: "There is not enough evidence to decide.",
  };
  const verdictChip = (v) => `<span class="chip v-${esc(v)}">${esc(LABELS[v] || v)}</span>`;
  const judgeChip = (j) => `<span class="chip j-${esc(j)}">${esc(LABELS[j] || j)}</span>`;

  // Real SciFact dev claims. The last one is a known failure, kept on purpose.
  const EXAMPLES = [
    { claim: "Monoclonal antibody targeting of N-cadherin inhibits metastasis.", gold: "SUPPORTED" },
    { claim: "Neutrophil extracellular traps (NETs) are released by ANCA-stimulated neutrophils.", gold: "SUPPORTED" },
    { claim: "Chenodeosycholic acid treatment reduces whole-body energy expenditure.", gold: "CONTRADICTED" },
    { claim: "ALDH1 expression is associated with better breast cancer outcomes.", gold: "CONTRADICTED" },
    { claim: "Flexible molecules experience greater steric hindrance in the tumor microenviroment than rigid molecules.", gold: "UNKNOWN" },
    { claim: "ALDH1 expression is associated with poorer prognosis in breast cancer.", gold: "SUPPORTED", note: "the system gets this one wrong" },
  ];

  // ------------------------------------------------------------------ state
  const DEFAULTS = { topK: 3, thr: 0.5 };
  const state = {
    topK: store.get("sv-topk", DEFAULTS.topK),
    thr: store.get("sv-thr", DEFAULTS.thr),
    judge: store.get("sv-judge", null),
    judges: [],
    status: null,
    controller: null,
    current: null,
    history: store.get("sv-history", []),
  };

  // ------------------------------------------------------------------ theme
  function currentTheme() {
    const set = document.documentElement.dataset.theme;
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("sv-theme", next); } catch (e) { /* ignore */ }
  }
  $("#theme-toggle").addEventListener("click", toggleTheme);

  // ------------------------------------------------------------------ router
  const ROUTES = ["verify", "batch", "library", "benchmarks", "method"];
  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    const [path, query = ""] = raw.split("?");
    return { route: ROUTES.includes(path) ? path : "verify", params: new URLSearchParams(query) };
  }
  function show(route) {
    $$(".view").forEach((v) => { v.hidden = v.dataset.view !== route; });
    $$(".tabs a").forEach((a) => {
      if (a.dataset.route === route) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (route === "benchmarks") loadBenchmarks();
    if (route === "library") setTimeout(() => $("#search-q").focus(), 30);
  }
  function onRoute() {
    const { route, params } = parseHash();
    show(route);
    if (route === "verify" && params.get("claim")) {
      if (params.get("k")) setTopK(+params.get("k"));
      if (params.get("t")) setThr(+params.get("t"));
      $("#claim").value = params.get("claim");
      updateHints();
      history.replaceState(null, "", "#/verify");
      run();
    }
  }
  window.addEventListener("hashchange", onRoute);
  const go = (route) => { if (location.hash !== "#/" + route) location.hash = "#/" + route; else show(route); };

  // ------------------------------------------------------------------ model status
  let statusTimer;
  async function pollStatus() {
    clearTimeout(statusTimer);
    const el = $("#status"), text = $(".status-text", el);
    try {
      const s = await api("/status");
      // The status pill describes the judge currently selected, not just the default one.
      const mine = (s.judges || []).find((j) => j.key === state.judge);
      if (mine) {
        s.model_ready = mine.ready; s.loading = mine.loading; s.error = mine.error;
        state.judges = s.judges;
      }
      state.status = s;
      const model = mine ? shortJudge(mine.name) : (s.nli_model || "").split("/").pop();
      if (s.model_ready) {
        el.dataset.state = "ready";
        text.textContent = `Ready · ${model}`;
        el.title = `${mine ? mine.path : s.nli_model} loaded · ${s.corpus_size?.toLocaleString()} abstracts · ${s.cached_results} cached results`;
      } else if (s.error) {
        el.dataset.state = "error";
        text.textContent = "Model failed to load";
        el.title = s.error;
      } else {
        el.dataset.state = "loading";
        text.textContent = `Loading ${model}…`;
        el.title = "Loading the corpus and the language model. First start can take a minute.";
      }
      $$("[data-model]").forEach((n) => { n.textContent = (s.nli_model || "").split("/").pop() || n.textContent; });
      statusTimer = setTimeout(pollStatus, s.model_ready || s.error ? 20000 : 1500);
    } catch (e) {
      el.dataset.state = "error";
      text.textContent = "Server offline";
      el.title = "Start it with: uvicorn api.main:app --port 8000";
      statusTimer = setTimeout(pollStatus, 4000);
    }
  }

  // ------------------------------------------------------------------ settings
  function setTopK(v) {
    state.topK = Math.min(10, Math.max(1, Math.round(v) || DEFAULTS.topK));
    $("#topk").value = state.topK;
    $("#topk-out").textContent = state.topK;
    store.set("sv-topk", state.topK);
  }
  function setThr(v) {
    state.thr = Math.min(0.99, Math.max(0, isFinite(v) ? v : DEFAULTS.thr));
    $("#thr").value = state.thr;
    $("#thr-out").textContent = state.thr.toFixed(2);
    store.set("sv-thr", state.thr);
  }
  $("#topk").addEventListener("input", (e) => setTopK(+e.target.value));
  $("#thr").addEventListener("input", (e) => setThr(+e.target.value));
  $("#settings-reset").addEventListener("click", () => { setTopK(DEFAULTS.topK); setThr(DEFAULTS.thr); });
  $("#settings-btn").addEventListener("click", (e) => {
    const panel = $("#settings");
    panel.hidden = !panel.hidden;
    e.currentTarget.setAttribute("aria-expanded", String(!panel.hidden));
  });
  const requestOptions = () => ({ top_k: state.topK, min_confidence: state.thr, model: state.judge || undefined });

  // ------------------------------------------------------------------ judges
  const judgeByKey = (key) => state.judges.find((j) => j.key === key);
  const shortJudge = (name) => String(name || "").replace(/\s*\(.*\)\s*$/, "");
  function renderJudgeSelect() {
    const sel = $("#judge");
    sel.innerHTML = state.judges.map((j) => `<option value="${esc(j.key)}" title="${esc(j.description)}">${esc(j.name)}${j.calibrated ? " · calibrated" : ""}</option>`).join("");
    sel.value = state.judge;
    sel.title = (judgeByKey(state.judge) || {}).description || "";
  }
  async function loadJudges() {
    try {
      const data = await api("/judges");
      state.judges = data.judges;
      if (!judgeByKey(state.judge)) state.judge = data.default;
      store.set("sv-judge", state.judge);
      renderJudgeSelect();
      $("#judge").closest(".judge-pick").hidden = state.judges.length < 2;
    } catch (e) { $("#judge").closest(".judge-pick").hidden = true; }
  }
  $("#judge").addEventListener("change", (e) => {
    state.judge = e.target.value;
    store.set("sv-judge", state.judge);
    renderJudgeSelect();
    api(`/warmup?model=${encodeURIComponent(state.judge)}`, { method: "POST" }).catch(() => {});
    pollStatus();
  });

  // ------------------------------------------------------------------ composer
  const claimEl = $("#claim");
  function updateHints() {
    const text = claimEl.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const counter = $("#counter");
    counter.textContent = `${claimEl.value.length} / 1000`;
    counter.classList.toggle("over", claimEl.value.length > 1000);
    const hints = [];
    if (words) hints.push(`${words} word${words === 1 ? "" : "s"}`);
    if (text.endsWith("?")) hints.push('<span class="warn">Phrase it as a statement, not a question.</span>');
    else if (words && words < 4) hints.push('<span class="warn">Very short claims match fewer papers.</span>');
    if (claimEl.value.length > 1000) hints.push('<span class="warn">Only the first 1,000 characters will be checked.</span>');
    $("#claim-hints").innerHTML = hints.join("");
    $("#examples").hidden = Boolean(text) || !$("#result").hidden;
  }
  claimEl.addEventListener("input", updateHints);
  claimEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
  });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); run(); });

  function renderExamples() {
    $("#example-cards").innerHTML = EXAMPLES.map((ex, i) => `
      <button type="button" class="example-card" data-i="${i}">
        <q>${esc(ex.claim)}</q>
        <span class="ex-meta">Dataset says ${verdictChip(ex.gold)}${ex.note ? ` · ${esc(ex.note)}` : ""}</span>
      </button>`).join("");
    $$(".example-card").forEach((b) => b.addEventListener("click", () => {
      claimEl.value = EXAMPLES[+b.dataset.i].claim;
      updateHints();
      run();
    }));
  }

  async function randomClaim() {
    const label = $("#random-label").value;
    try {
      const data = await api(`/examples?n=1${label ? "&label=" + label : ""}`);
      if (!data.claims.length) { toast("The dataset is not available on this machine."); return; }
      claimEl.value = data.claims[0].claim;
      updateHints();
      claimEl.focus();
      toast(`Dataset says: ${LABELS[data.claims[0].gold_verdict]}. Press Verify to compare.`);
    } catch (e) { toast(e.message); }
  }
  $("#random-btn").addEventListener("click", randomClaim);

  // ------------------------------------------------------------------ verify (streaming)
  const STEPS = ["load", "search", "judge", "weigh"];
  function station(step, cls) {
    const li = $(`.stations li[data-step="${step}"]`);
    li.classList.remove("active", "done", "skipped");
    if (cls) li.classList.add(cls);
  }

  let tick;
  function startClock() {
    const t0 = performance.now();
    clearInterval(tick);
    tick = setInterval(() => { $("#elapsed").textContent = fmtTime((performance.now() - t0) / 1000); }, 100);
  }

  async function run() {
    const claim = claimEl.value.trim();
    if (!claim) { claimEl.focus(); toast("Type a claim first."); return; }
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    state.current = null;

    $("#error-box").hidden = true;
    $("#examples").hidden = true;
    $("#verify-btn").disabled = true;
    $("#progress").hidden = false;
    STEPS.forEach((s) => station(s, null));
    const modelReady = state.status && state.status.model_ready;
    station("load", modelReady ? "skipped" : "active");
    if (modelReady) station("search", "active");
    startClock();

    const result = $("#result");
    result.hidden = false;
    result.innerHTML = `
      <div class="verdict-panel" style="--vc:var(--rule-strong)">
        <div class="stamp" style="--vc:var(--muted);animation:none;opacity:.5"><small>Verdict</small>…</div>
        <div><p class="verdict-claim">“${esc(claim)}”</p><div class="skeleton" style="width:60%"></div></div>
      </div>
      <div class="section-title"><h2>Evidence</h2></div>
      <ol class="ledger" id="ledger"></ol>`;

    let judged = 0, total = 0, live = { evidence: [], claim_terms: [] };
    try {
      await streamVerify({ claim, ...requestOptions() }, controller.signal, (ev) => {
        if (ev.event === "loading") { station("load", "active"); }
        else if (ev.event === "started") {
          station("load", $(`.stations li[data-step="load"]`).classList.contains("active") ? "done" : "skipped");
          station("search", "active");
        } else if (ev.event === "retrieved") {
          if (!$(`.stations li[data-step="load"]`).classList.contains("done")) station("load", "skipped");
          station("search", "done");
          total = ev.papers.length;
          $("#judge-sub").textContent = `0 / ${total}`;
          station("judge", total ? "active" : "skipped");
          if (!total) station("weigh", "active");
          $("#ledger").innerHTML = ev.papers.map((p, i) => `
            <li class="paper pending" data-doc="${p.doc_id}">
              <div class="paper-rank">${i + 1}</div>
              <div><h3 class="paper-title">${esc(p.title)}</h3>
                <div class="skeleton" style="width:90%"></div><div class="skeleton" style="width:70%"></div></div>
            </li>`).join("");
        } else if (ev.event === "judged") {
          judged += 1;
          live.evidence.push(ev.item);
          $("#judge-sub").textContent = `${judged} / ${total}`;
          const li = $(`#ledger li[data-doc="${ev.item.doc_id}"]`);
          if (li) li.outerHTML = paperCard(ev.item, judged - 1, { claim, terms: [], gold: [], maxScore: live.evidence[0].retrieval_score });
          if (judged === total) { station("judge", "done"); station("weigh", "active"); }
        } else if (ev.event === "done") {
          STEPS.forEach((s) => { if (!$(`.stations li[data-step="${s}"]`).classList.contains("skipped")) station(s, "done"); });
          showResult(ev.result, { fresh: true });
        } else if (ev.event === "error") {
          throw new Error(ev.detail);
        }
      });
    } catch (e) {
      if (e.name === "AbortError") { result.hidden = true; toast("Cancelled."); }
      else { result.hidden = true; showError(e.message); }
    } finally {
      clearInterval(tick);
      if (state.controller === controller) state.controller = null;
      $("#verify-btn").disabled = false;
      $("#progress").hidden = true;
      pollStatus();
    }
  }

  async function streamVerify(body, signal, onEvent) {
    const res = await fetch("/verify/stream", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    });
    if (!res.ok) throw new Error(await errorText(res));
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer));
  }

  $("#cancel-btn").addEventListener("click", () => { if (state.controller) state.controller.abort(); });

  function showError(msg) {
    const box = $("#error-box");
    box.innerHTML = `<b>Something went wrong.</b> ${esc(msg)}`;
    box.hidden = false;
  }

  // ------------------------------------------------------------------ result rendering
  function showResult(r, { fresh = false } = {}) {
    state.current = r;
    const result = $("#result");
    result.hidden = false;
    $("#examples").hidden = true;
    const ctx = {
      claim: r.claim, terms: r.claim_terms || [], gold: (r.gold && r.gold.gold_doc_ids) || [],
      maxScore: Math.max(1e-9, ...r.evidence.map((e) => e.retrieval_score)), thr: r.settings?.min_confidence ?? state.thr,
    };
    const vc = `var(--${r.verdict})`;

    let answer = "";
    if (r.gold) {
      const match = r.gold.gold_verdict === r.verdict;
      answer = `<div class="stat"><span class="stat-k">Dataset answer</span>
        <span class="answer-key">${verdictChip(r.gold.gold_verdict)}
        <span class="${match ? "ok" : "bad"}">${match ? "✓ match" : "✗ differs"}</span></span></div>`;
    }

    result.innerHTML = `
      <div class="verdict-panel" style="--vc:${vc}">
        <div class="stamp" style="--vc:${vc}"><small>Verdict</small>${esc(LABELS[r.verdict] || r.verdict)}</div>
        <div>
          <p class="verdict-claim">“${esc(r.claim)}”</p>
          <p class="verdict-why"><b>${esc(HEADLINES[r.verdict] || "")}</b> ${esc(r.explanation)}</p>
        </div>
        <div class="verdict-stats" style="--vc:${vc}">
          <div class="stat"><span class="stat-k">Model confidence</span>
            <span class="stat-v">${r.confidence.toFixed(2)}</span>
            <span class="meter"><i style="width:${pct(r.confidence)}"></i></span></div>
          <div class="stat"><span class="stat-k">Papers judged</span><span class="stat-v">${r.evidence.length}</span></div>
          <div class="stat"><span class="stat-k">Threshold</span><span class="stat-v">${(ctx.thr).toFixed(2)}</span></div>
          ${r.confidence_calibrated != null ? `<div class="stat" title="After temperature scaling fitted on the training split: a number meant to match how often verdicts like this one are right. The raw value on the left is what the evaluation used."><span class="stat-k">Calibrated</span><span class="stat-v">${r.confidence_calibrated.toFixed(2)}</span><span class="meter"><i style="width:${pct(r.confidence_calibrated)}"></i></span></div>` : ""}
          <div class="stat"><span class="stat-k">Time</span><span class="stat-v">${fmtTime(r.elapsed_seconds)}</span></div>
          ${r.judge ? `<div class="stat" title="${esc(r.judge)}"><span class="stat-k">Judge</span><span class="stat-v judge-name">${esc(shortJudge(r.judge))}</span></div>` : ""}
          ${answer}
          ${r.cached ? '<span class="cached-tag" title="Same claim and settings as an earlier check, answered from memory">cached</span>' : ""}
        </div>
      </div>
      <div class="extras" id="extras"></div>
      ${r.evidence.length ? scale(r, ctx) : ""}
      <div class="section-title">
        <h2>Evidence</h2>
        <span class="eyebrow">${r.evidence.length ? "ranked by search score" : ""}</span>
      </div>
      ${r.evidence.length
        ? `<ol class="ledger" id="ledger">${r.evidence.map((e, i) => paperCard(e, i, ctx)).join("")}</ol>`
        : `<p class="empty-note">No paper in the corpus shares meaningful vocabulary with this claim. The library only covers biomedical research, so try the <a href="#/library">Library</a> to see what it contains.</p>`}
      <div class="result-actions">
        ${r.evidence.length ? `<button class="btn" type="button" data-act="counter" title="Build the opposite of this claim and verify it too. A judge that supports both is not reading the evidence.">
          <svg viewBox="0 0 24 24"><path d="M4 7h13M13 3l4 4-4 4M20 17H7M11 13l-4 4 4 4"/></svg>Check the opposite</button>` : ""}
        ${state.judges.length > 1 ? `<button class="btn" type="button" data-act="compare" title="Run the same claim through every judge">
          <svg viewBox="0 0 24 24"><path d="M4 6h7v12H4zM13 6h7v12h-7z"/></svg>Compare judges</button>` : ""}
        <button class="btn" type="button" data-act="share">
          <svg viewBox="0 0 24 24"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7L11.5 6.8M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1.5-1.5"/></svg>Copy link</button>
        <button class="btn" type="button" data-act="md">
          <svg viewBox="0 0 24 24"><path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19h14"/></svg>Report (.md)</button>
        <button class="btn" type="button" data-act="json">JSON</button>
        <button class="btn" type="button" data-act="print">
          <svg viewBox="0 0 24 24"><path d="M7 9V4h10v5M7 17H5a1 1 0 0 1-1-1v-5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v5a1 1 0 0 1-1 1h-2M7 14h10v6H7z"/></svg>Print / PDF</button>
        <button class="btn btn-ghost" type="button" data-act="new">New claim</button>
      </div>
      <p class="caveat">${esc(r.caveat)}</p>`;

    state.ctx = ctx;
    if (fresh) addHistory(r);
    updateHints();
  }

  function scale(r, ctx) {
    const segs = r.evidence.map((e, i) => {
      const weak = e.judgement !== "NEUTRAL" && e.nli_score < ctx.thr;
      return `<a class="scale-seg s-${e.judgement}${weak ? " weak" : ""}" href="#paper-${e.doc_id}" data-doc="${e.doc_id}"
        style="flex:1" title="Paper ${i + 1}: ${esc(LABELS[e.judgement])} (${pct(e.nli_score)})${weak ? " · below threshold, not counted" : ""}">${i + 1}</a>`;
    }).join("");
    return `<div class="scale">
      <p class="eyebrow">Weight of evidence</p>
      <div class="scale-bar">${segs}</div>
      <div class="scale-legend">
        <span style="--c:var(--SUPPORTED)">supports</span>
        <span style="--c:var(--CONTRADICTED)">contradicts</span>
        <span style="--c:var(--NEUTRAL)">neutral</span>
        <span>striped = below threshold, not counted</span>
      </div></div>`;
  }

  function paperCard(e, i, ctx) {
    const probs = e.probabilities && Object.keys(e.probabilities).length ? e.probabilities
      : { [e.nli_label]: e.nli_score };
    const pE = probs.entailment || 0, pN = probs.neutral || 0, pC = probs.contradiction || 0;
    const isGold = ctx.gold.includes(e.doc_id);
    const shown = new Set(e.highlighted || []);
    const sentences = e.sentences && e.sentences.length ? e.sentences : [e.passage];
    let prevShown = false, body = "";
    sentences.forEach((s, idx) => {
      const on = shown.has(idx) || !e.sentences || !e.sentences.length;
      if (!on && (idx === 0 || prevShown)) body += '<span class="gap"> … </span>';
      body += `<span class="sent${on ? " hl" : " dim"}" data-i="${idx}">${markTerms(s, ctx.terms)} </span>`;
      prevShown = on;
    });
    const rel = e.retrieval_score / ctx.maxScore;
    return `
      <li class="paper" id="paper-${e.doc_id}" data-doc="${e.doc_id}" style="animation-delay:${i * 60}ms">
        <div class="paper-rank">${i + 1}</div>
        <div>
          <div class="paper-head">
            <h3 class="paper-title"><button type="button" data-open="${e.doc_id}">${esc(e.title)}</button></h3>
            <div class="paper-badges">
              ${judgeChip(e.judgement)}
              ${isGold ? '<span class="chip gold" title="The SciFact annotators marked this paper as the evidence for this claim">Annotators’ evidence</span>' : ""}
            </div>
          </div>
          <div class="paper-meta">
            <div class="stat"><span class="stat-k">Search match</span>
              <span class="stat-v">${e.retrieval_score.toFixed(1)}</span>
              <span class="meter" style="--vc:var(--accent)"><i style="width:${pct(rel)}"></i></span></div>
            <div class="nli">
              <span class="stat-k">Judge's reading</span>
              <div class="nli-bar" title="Entails ${pct(pE, 1)} · Neutral ${pct(pN, 1)} · Contradicts ${pct(pC, 1)}">
                <i class="e" style="width:${pct(pE, 2)}"></i><i class="n" style="width:${pct(pN, 2)}"></i><i class="c" style="width:${pct(pC, 2)}"></i>
              </div>
              <div class="nli-legend"><span>entails <b>${pct(pE)}</b></span><span>neutral <b>${pct(pN)}</b></span><span>contradicts <b>${pct(pC)}</b></span></div>
            </div>
            <div class="stat"><span class="stat-k">Paper ID</span><span class="stat-v">${e.doc_id}</span></div>
          </div>
          <p class="abstract">${body}</p>
          <div class="paper-actions">
            ${e.sentences && e.sentences.length > shown.size ? '<button class="link-btn" type="button" data-expand>Show full abstract</button>' : ""}
            <button class="link-btn" type="button" data-explain="${e.doc_id}" title="Judge every sentence on its own to find the ones that decide it">Find deciding sentences</button>
            <span class="sent-legend" data-legend></span>
          </div>
        </div>
      </li>`;
  }

  // One delegated listener for everything inside the result, bound once.
  $("#result").addEventListener("click", (ev) => {
      const t = ev.target.closest("button, a");
      const r = state.current, ctx = state.ctx;
      if (!t || !r || !ctx) return;
      if (t.hasAttribute("data-close-extra")) {
        t.closest(".extra").remove();
      } else if (t.matches(".scale-seg")) {
        ev.preventDefault();
        const card = $(`#paper-${t.dataset.doc}`);
        card.scrollIntoView({ behavior: "smooth", block: "start" });
        card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
      } else if (t.dataset.open) {
        openPaper(+t.dataset.open, ctx.terms);
      } else if (t.hasAttribute("data-expand")) {
        const abs = $(".abstract", t.closest(".paper"));
        abs.classList.toggle("expanded");
        t.textContent = abs.classList.contains("expanded") ? "Show matching sentences only" : "Show full abstract";
      } else if (t.dataset.explain) {
        explain(t, r.claim, +t.dataset.explain, ctx);
      } else if (t.dataset.act) {
        resultAction(t.dataset.act, r);
      }
  });

  async function explain(btn, claim, docId, ctx) {
    const card = btn.closest(".paper");
    btn.disabled = true;
    btn.textContent = "Reading each sentence…";
    try {
      const data = await api("/explain", { method: "POST", body: JSON.stringify({ claim, doc_id: docId }) });
      const abs = $(".abstract", card);
      abs.classList.add("expanded");
      const expandBtn = $("[data-expand]", card);
      if (expandBtn) expandBtn.remove();
      let sup = 0, con = 0;
      data.sentences.forEach((s) => {
        const span = $(`.sent[data-i="${s.index}"]`, abs);
        if (!span) return;
        span.classList.remove("hl");
        $$(".sent-tag", span).forEach((n) => n.remove());
        if (s.judgement !== "NEUTRAL" && s.score >= ctx.thr) {
          span.dataset.j = s.judgement;
          span.title = `${LABELS[s.judgement]} the claim on its own (${pct(s.score, 1)})`;
          span.insertAdjacentHTML("beforeend", `<span class="sent-tag t-${s.judgement}">${s.judgement === "SUPPORT" ? "+" : "−"}${pct(s.score)}</span>`);
          if (s.judgement === "SUPPORT") sup += 1; else con += 1;
        }
      });
      $("[data-legend]", card).textContent = sup + con
        ? `${sup} sentence${sup === 1 ? " supports" : "s support"} · ${con} contradict${con === 1 ? "s" : ""}, each read on its own`
        : "No single sentence decides it on its own; the verdict comes from the abstract as a whole.";
      btn.remove();
    } catch (e) {
      btn.disabled = false;
      btn.textContent = "Find deciding sentences";
      toast(e.message);
    }
  }

  function shareLink(r) {
    const q = new URLSearchParams({ claim: r.claim, k: r.settings?.top_k ?? state.topK, t: r.settings?.min_confidence ?? state.thr });
    return `${location.origin}${location.pathname}#/verify?${q}`;
  }

  function markdownReport(r) {
    const lines = [
      `# SciVerify report`, ``,
      `**Claim:** ${r.claim}`, ``,
      `**Verdict:** ${r.verdict}  (model confidence ${r.confidence.toFixed(2)}, not calibrated)`, ``,
      r.explanation, ``,
    ];
    if (r.gold) lines.push(`**Dataset answer:** ${r.gold.gold_verdict} (${r.gold.gold_verdict === r.verdict ? "match" : "differs"})`, ``);
    lines.push(`## Evidence`, ``);
    r.evidence.forEach((e, i) => {
      const p = e.probabilities || {};
      lines.push(`### ${i + 1}. ${e.title}`, ``,
        `- Paper ID: ${e.doc_id}`,
        `- Search score: ${e.retrieval_score}`,
        `- Judge: ${e.judgement} (entails ${pct(p.entailment || 0)}, neutral ${pct(p.neutral || 0)}, contradicts ${pct(p.contradiction || 0)})`,
        ``, `> ${e.passage}`, ``);
    });
    lines.push(`---`, `Settings: top ${r.settings?.top_k} papers, threshold ${r.settings?.min_confidence}. Time ${r.elapsed_seconds}s.`, ``, `_${r.caveat}_`, ``);
    return lines.join("\n");
  }

  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 50).replace(/^-|-$/g, "");

  const STATUS_LABEL = { consistent: "Consistent", inconsistent: "Unreliable", inconclusive: "Inconclusive" };

  function markChanged(text, changed) {
    if (!changed) return esc(text);
    const to = changed[1];
    const i = text.toLowerCase().indexOf(to.toLowerCase());
    if (i < 0) return esc(text);
    return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + to.length)) + "</mark>" + esc(text.slice(i + to.length));
  }

  async function checkOpposite(r) {
    const extras = $("#extras");
    let box = $("#extra-counter");
    if (!box) {
      box = document.createElement("div");
      box.className = "extra"; box.id = "extra-counter";
      extras.appendChild(box);
    }
    box.innerHTML = `<div class="extra-head"><h3>Opposite-claim check</h3><span class="eyebrow">verifying the opposite…</span></div><div class="skeleton" style="width:70%"></div>`;
    try {
      const c = await api("/counterclaim", { method: "POST", body: JSON.stringify({ claim: r.claim, ...requestOptions(), model: r.judge_key || state.judge }) });
      const how = { antonym: `swapped “${c.changed?.[0]}” for “${c.changed?.[1]}”`, negate: `inserted “${c.changed?.[1]}”`,
        "remove-negation": `removed “${c.changed?.[0]}”`, prefix: "prefixed “It is not the case that”" }[c.method] || c.method;
      box.innerHTML = `
        <div class="extra-head"><h3>Opposite-claim check</h3><span class="badge ${esc(c.status)}">${esc(STATUS_LABEL[c.status] || c.status)}</span>
          <button class="link-btn" type="button" data-close-extra>Hide</button></div>
        <div class="pair">
          <div class="pair-side"><p class="eyebrow">The claim</p><q>${esc(c.claim)}</q>
            <div class="mini">${verdictChip(c.verdict)} <span>${c.confidence.toFixed(2)}</span></div></div>
          <div class="pair-side"><p class="eyebrow">Its opposite (${esc(how)})</p><q>${markChanged(c.counter_claim, c.changed)}</q>
            <div class="mini">${verdictChip(c.counter_verdict)} <span>${c.counter_confidence.toFixed(2)}</span></div></div>
        </div>
        <p class="note${c.status === "inconsistent" ? " warn" : ""}">${esc(c.note)}</p>`;
    } catch (e) {
      box.innerHTML = `<div class="extra-head"><h3>Opposite-claim check</h3><button class="link-btn" type="button" data-close-extra>Hide</button></div><p class="note">${esc(e.message)}</p>`;
    }
  }

  async function compareJudges(r) {
    const extras = $("#extras");
    let box = $("#extra-compare");
    if (!box) {
      box = document.createElement("div");
      box.className = "extra"; box.id = "extra-compare";
      extras.appendChild(box);
    }
    const rows = state.judges.map((j) => ({ j, result: j.key === (r.judge_key || state.judge) ? r : null, pending: j.key !== (r.judge_key || state.judge) }));
    const render = () => {
      const verdicts = new Set(rows.filter((x) => x.result).map((x) => x.result.verdict));
      box.innerHTML = `
        <div class="extra-head"><h3>Every judge on this claim</h3>
          ${verdicts.size > 1 ? '<span class="badge inconsistent">judges disagree</span>' : rows.every((x) => x.result) ? '<span class="badge consistent">all agree</span>' : ""}
          <button class="link-btn" type="button" data-close-extra>Hide</button></div>
        <div class="table-wrap" style="margin:0;border:0"><table class="compare-table">
          <thead><tr><th>Judge</th><th>Verdict</th><th class="num">Confidence</th><th class="num">Calibrated</th><th class="num">Time</th></tr></thead>
          <tbody>${rows.map((x) => `<tr class="${x.result === r ? "current" : ""}${x.result && x.result.verdict !== r.verdict ? " disagree" : ""}">
            <td title="${esc(x.j.description)}"><b>${esc(shortJudge(x.j.name))}</b></td>
            <td>${x.result ? verdictChip(x.result.verdict) : x.pending ? '<span style="color:var(--muted)">loading judge, then reading papers…</span>' : ""}</td>
            <td class="num">${x.result ? x.result.confidence.toFixed(2) : ""}</td>
            <td class="num">${x.result && x.result.confidence_calibrated != null ? x.result.confidence_calibrated.toFixed(2) : x.result ? "—" : ""}</td>
            <td class="num">${x.result ? fmtTime(x.result.elapsed_seconds) : ""}</td></tr>`).join("")}</tbody></table></div>
        ${r.gold ? `<p class="note">Dataset answer: ${verdictChip(r.gold.gold_verdict)}</p>` : ""}`;
    };
    render();
    for (const x of rows) {
      if (x.result) continue;
      try {
        x.result = await api("/verify", { method: "POST", body: JSON.stringify({ claim: r.claim, ...requestOptions(), model: x.j.key }) });
      } catch (e) { x.result = null; x.pending = false; x.error = e.message; }
      render();
    }
  }

  function resultAction(act, r) {
    if (act === "counter") checkOpposite(r);
    else if (act === "compare") compareJudges(r);
    else if (act === "share") copy(shareLink(r), "Link copied. Opening it re-runs this check.");
    else if (act === "md") download(`sciverify-${slug(r.claim)}.md`, markdownReport(r), "text/markdown");
    else if (act === "json") download(`sciverify-${slug(r.claim)}.json`, JSON.stringify(r, null, 2), "application/json");
    else if (act === "print") window.print();
    else if (act === "new") {
      $("#result").hidden = true;
      claimEl.value = "";
      updateHints();
      claimEl.focus();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  // ------------------------------------------------------------------ history
  function addHistory(r) {
    state.history = [{ at: Date.now(), result: r }, ...state.history.filter((h) => !(h.result.claim === r.claim
      && h.result.judge === r.judge && JSON.stringify(h.result.settings) === JSON.stringify(r.settings)))].slice(0, 30);
    store.set("sv-history", state.history);
    renderHistory();
  }

  function ago(ts) {
    const s = (Date.now() - ts) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(ts).toLocaleDateString();
  }

  function renderHistory() {
    const list = $("#history-list");
    list.innerHTML = state.history.map((h, i) => `
      <li><button type="button" data-h="${i}" style="--vc:var(--${esc(h.result.verdict)})">
        <span class="h-dot"></span>
        <span class="h-claim">${esc(h.result.claim)}</span>
        <span class="h-meta">${esc(LABELS[h.result.verdict])} · ${h.result.confidence.toFixed(2)}${h.result.judge ? " · " + esc(shortJudge(h.result.judge)) : ""} · ${ago(h.at)}</span>
      </button></li>`).join("");
    $("#history-empty").hidden = state.history.length > 0;
    $("#history-clear").hidden = !state.history.length;
  }
  $("#history-list").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-h]");
    if (!b) return;
    const r = state.history[+b.dataset.h].result;
    claimEl.value = r.claim;
    showResult(r);
    $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("#history-clear").addEventListener("click", () => {
    state.history = [];
    store.set("sv-history", []);
    renderHistory();
  });

  // ------------------------------------------------------------------ paper reader
  const reader = $("#reader");
  async function openPaper(docId, terms = []) {
    $("#reader-id").textContent = `Paper ${docId}`;
    $("#reader-title").textContent = "Loading…";
    $("#reader-body").innerHTML = '<div class="skeleton"></div><div class="skeleton" style="width:80%"></div>';
    if (!reader.open) reader.showModal();
    try {
      const p = await api(`/papers/${docId}`);
      $("#reader-title").textContent = p.title;
      $("#reader-body").innerHTML = p.sentences.map((s, i) => `<p><span class="n">${i + 1}</span>${markTerms(s, terms)}</p>`).join("")
        || '<p class="empty-note">This paper has no abstract text.</p>';
      reader.dataset.title = p.title;
    } catch (e) {
      $("#reader-title").textContent = "Could not load this paper";
      $("#reader-body").innerHTML = `<p>${esc(e.message)}</p>`;
    }
  }
  $("#reader-close").addEventListener("click", () => reader.close());
  reader.addEventListener("click", (e) => { if (e.target === reader) reader.close(); });
  $("#reader-use").addEventListener("click", () => {
    reader.close();
    claimEl.value = reader.dataset.title || "";
    go("verify");
    updateHints();
    claimEl.focus();
  });

  // ------------------------------------------------------------------ batch
  const batch = { rows: [], gold: new Map(), running: false, stop: false };

  function parseCsv(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') quoted = false;
        else field += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += c;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows.filter((r) => r.some((f) => f.trim()));
  }

  $("#batch-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    let claims = [];
    if (/\.csv$/i.test(file.name)) {
      const rows = parseCsv(text);
      const header = rows[0].map((h) => h.trim().toLowerCase());
      let ci = header.indexOf("claim"), gi = header.findIndex((h) => ["label", "gold", "verdict", "gold_verdict"].includes(h));
      const hasHeader = ci >= 0;
      if (ci < 0) ci = 0;
      rows.slice(hasHeader ? 1 : 0).forEach((r) => {
        const c = (r[ci] || "").trim();
        if (!c) return;
        claims.push(c);
        if (gi >= 0 && r[gi]) batch.gold.set(c, r[gi].trim().toUpperCase());
      });
    } else {
      claims = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    }
    $("#batch-text").value = claims.join("\n");
    toast(`Imported ${claims.length} claim${claims.length === 1 ? "" : "s"}.`);
    e.target.value = "";
  });

  $("#batch-draw").addEventListener("click", async () => {
    const n = Math.min(50, Math.max(1, +$("#batch-n").value || 10));
    try {
      const data = await api(`/examples?n=${n}`);
      if (!data.claims.length) { toast("The dataset is not available on this machine."); return; }
      data.claims.forEach((c) => batch.gold.set(c.claim, c.gold_verdict));
      $("#batch-text").value = data.claims.map((c) => c.claim).join("\n");
    } catch (e) { toast(e.message); }
  });

  function batchSummary() {
    const done = batch.rows.filter((r) => r.result);
    const withGold = done.filter((r) => r.gold);
    const agree = withGold.filter((r) => r.gold === r.result.verdict).length;
    const counts = {};
    done.forEach((r) => { counts[r.result.verdict] = (counts[r.result.verdict] || 0) + 1; });
    const time = done.reduce((s, r) => s + r.result.elapsed_seconds, 0);
    const kpi = (v, k, frac) => `<div class="kpi"><div class="kpi-v">${v}</div><div class="kpi-k">${k}</div>${frac != null ? `<div class="kpi-bar"><i style="width:${pct(frac)}"></i></div>` : ""}</div>`;
    $("#batch-summary").innerHTML = [
      kpi(`${done.length}<span style="color:var(--muted);font-size:.6em"> / ${batch.rows.length}</span>`, "claims checked", batch.rows.length ? done.length / batch.rows.length : 0),
      withGold.length ? kpi(pct(agree / withGold.length), `agree with the answer key (${agree} of ${withGold.length})`, agree / withGold.length) : "",
      ...["SUPPORTED", "CONTRADICTED", "CONFLICTING", "UNKNOWN"].map((v) =>
        `<div class="kpi"><div class="kpi-v" style="color:var(--${v})">${counts[v] || 0}</div><div class="kpi-k">${LABELS[v]}</div></div>`),
      done.length ? kpi(fmtTime(time / done.length), "average per claim") : "",
    ].join("");
    $("#batch-summary").hidden = false;
  }

  function batchRow(r, i) {
    const res = r.result;
    let key = '<span style="color:var(--muted)">—</span>';
    if (r.gold) key = `${verdictChip(r.gold)} ${res ? (r.gold === res.verdict ? '<span class="ok" style="color:var(--SUPPORTED)">✓</span>' : '<span style="color:var(--CONTRADICTED)">✗</span>') : ""}`;
    return `<tr data-row="${i}" class="${r.running ? "running" : ""}">
      <td class="num">${i + 1}</td>
      <td class="claim-cell">${res ? `<button type="button" data-open-row="${i}">${esc(r.claim)}</button>` : esc(r.claim)}</td>
      <td>${res ? verdictChip(res.verdict) : r.error ? `<span style="color:var(--CONTRADICTED)" title="${esc(r.error)}">error</span>` : r.running ? "checking…" : '<span style="color:var(--muted)">queued</span>'}</td>
      <td class="num">${res ? res.confidence.toFixed(2) : ""}</td>
      <td style="white-space:nowrap">${key}</td>
      <td class="num">${res ? fmtTime(res.elapsed_seconds) : ""}</td></tr>`;
  }
  function renderBatch() {
    $("#batch-table tbody").innerHTML = batch.rows.map(batchRow).join("");
    batchSummary();
  }

  $("#batch-run").addEventListener("click", async () => {
    const claims = $("#batch-text").value.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (!claims.length) { toast("Add at least one claim, one per line."); return; }
    batch.rows = claims.slice(0, 200).map((c) => ({ claim: c, gold: batch.gold.get(c) || null }));
    batch.running = true; batch.stop = false;
    $("#batch-run").disabled = true; $("#batch-stop").hidden = false;
    $("#batch-table-wrap").hidden = false; $("#batch-export").hidden = true;
    renderBatch();
    for (let i = 0; i < batch.rows.length && !batch.stop; i++) {
      batch.rows[i].running = true;
      renderBatch();
      try {
        batch.rows[i].result = await api("/verify", { method: "POST", body: JSON.stringify({ claim: batch.rows[i].claim, ...requestOptions() }) });
      } catch (e) { batch.rows[i].error = e.message; }
      batch.rows[i].running = false;
      renderBatch();
    }
    batch.running = false;
    $("#batch-run").disabled = false; $("#batch-stop").hidden = true; $("#batch-export").hidden = false;
    toast(batch.stop ? "Batch stopped." : "Batch finished.");
  });
  $("#batch-stop").addEventListener("click", () => { batch.stop = true; });
  $("#batch-table").addEventListener("click", (e) => {
    const b = e.target.closest("[data-open-row]");
    if (!b) return;
    const r = batch.rows[+b.dataset.openRow].result;
    claimEl.value = r.claim;
    go("verify");
    showResult(r);
  });

  const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  $("#batch-csv").addEventListener("click", () => {
    const head = ["claim", "verdict", "confidence", "dataset_answer", "match", "papers", "seconds"];
    const lines = [head.join(",")].concat(batch.rows.map((r) => {
      const res = r.result || {};
      return [r.claim, res.verdict || (r.error ? "ERROR" : ""), res.confidence ?? "", r.gold || "",
        r.gold && res.verdict ? r.gold === res.verdict : "", (res.evidence || []).map((e) => e.doc_id).join(" "),
        res.elapsed_seconds ?? ""].map(csvCell).join(",");
    }));
    download("sciverify-batch.csv", lines.join("\n"), "text/csv");
  });
  $("#batch-json").addEventListener("click", () => {
    download("sciverify-batch.json", JSON.stringify(batch.rows.map((r) => ({ claim: r.claim, dataset_answer: r.gold, result: r.result || null, error: r.error || null })), null, 2), "application/json");
  });

  // ------------------------------------------------------------------ library
  $("#search-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#search-q").value.trim();
    const list = $("#search-results");
    if (!q) return;
    list.innerHTML = '<li><div class="skeleton"></div><div class="skeleton" style="width:70%"></div></li>';
    try {
      const data = await api(`/search?q=${encodeURIComponent(q)}&k=20`);
      const terms = q.toLowerCase().match(/[a-z0-9]+/g) || [];
      if (!data.results.length) { list.innerHTML = '<p class="empty-note">No abstract shares a word with that search.</p>'; return; }
      const top = data.results[0].score;
      list.innerHTML = data.results.map((r) => `
        <li>
          <button class="r-title" type="button" data-open="${r.doc_id}">${markTerms(r.title, terms)}</button>
          <p class="r-snippet">${markTerms(r.snippet, terms)}</p>
          <div class="r-score">score ${r.score.toFixed(1)}
            <span class="meter" style="--vc:var(--accent);width:100%"><i style="width:${pct(r.score / top)}"></i></span>
            <span class="r-id">ID ${r.doc_id}</span></div>
        </li>`).join("");
      list.dataset.terms = JSON.stringify(terms);
    } catch (err) { list.innerHTML = `<p class="empty-note">${esc(err.message)}</p>`; }
  });
  $("#search-results").addEventListener("click", (e) => {
    const b = e.target.closest("[data-open]");
    if (b) openPaper(+b.dataset.open, JSON.parse($("#search-results").dataset.terms || "[]"));
  });

  // ------------------------------------------------------------------ benchmarks
  let benchLoaded = false;
  async function loadBenchmarks() {
    if (benchLoaded) return;
    const root = $("#bench");
    root.innerHTML = '<div class="skeleton"></div>';
    try {
      const d = await api("/benchmarks");
      root.innerHTML = renderBenchmarks(d);
      benchLoaded = true;
    } catch (e) { root.innerHTML = `<p class="empty-note">Could not load results: ${esc(e.message)}</p>`; }
  }

  function renderBenchmarks(d) {
    const C = window.Charts;
    const base = d.baseline_c, bm25 = d.bm25_baseline, dense = d.dense_baseline, ea = d.error_analysis;
    const hybrid = d["proposed_hybrid_nosel_bart-large-mnli"];
    const panels = [];
    const kpis = [];
    const kpi = (v, k, sub) => `<div class="kpi"><div class="kpi-v">${v}</div><div class="kpi-k">${k}</div>${sub ? `<div class="kpi-k" style="margin-top:2px">${sub}</div>` : ""}</div>`;

    if (bm25) kpis.push(kpi(pct(bm25.recall_at_k["10"], 1), "right paper in the top 10", "BM25 search, dev claims with evidence"));
    if (base) {
      kpis.push(kpi(pct(base.verification.accuracy, 1), "verdicts match the answer key", `${base.num_dev_claims_total} dev claims`));
      kpis.push(kpi(base.verification.macro_f1_gold_classes.toFixed(2), "macro-F1", "over the classes that occur"));
    }
    if (ea && ea.judge_on_gold_documents) {
      const j = ea.judge_on_gold_documents;
      kpis.push(kpi(pct(j.judge_right / j.gold_doc_retrieved, 1), "judge reads the key paper right", `${j.judge_right} of ${j.gold_doc_retrieved}`));
    }

    if (bm25 && dense) {
      const series = [
        { name: "BM25", color: "var(--ink)", values: ["1", "5", "10"].map((k) => bm25.recall_at_k[k]) },
        { name: "Dense (MiniLM)", color: "var(--accent)", values: ["1", "5", "10"].map((k) => dense.recall_at_k[k]) },
      ];
      if (hybrid) series.push({ name: "Hybrid (RRF)", color: "var(--CONFLICTING)", values: ["1", "5", "10"].map((k) => hybrid.retrieval_metrics.recall_at_k[k]) });
      panels.push(`<div class="panel"><h2>Finding the right paper</h2>
        <p class="sub">Share of claims whose annotated evidence paper is in the top k results.</p>
        ${C.groupedBars({ categories: ["Recall@1", "Recall@5", "Recall@10"], series })}
        <div class="legend">${series.map((s) => `<span style="--c:${s.color}">${s.name} · MRR ${(s.name === "BM25" ? bm25.mrr : s.name.startsWith("Dense") ? dense.mrr : hybrid.retrieval_metrics.mrr).toFixed(3)}</span>`).join("")}</div></div>`);
    }

    const judgeName = (model) => {
      const m = String(model || "");
      if (/bart-large-mnli/.test(m)) return "General judge (bart-large)";
      if (/nli-scifact/.test(m)) return "Science-trained judge (MiniLM)";
      if (/MiniLM/i.test(m)) return "Small general judge (MiniLM)";
      return m.split("/").pop();
    };
    const judgeRuns = Object.entries(d)
      .filter(([k, v]) => /^proposed_bm25_nosel_/.test(k) && !/_train$|_limit\d+$/.test(k) && v.verification && v.num_claims === 300)
      .map(([k, v]) => ({ key: k, model: v.nli_model, name: judgeName(v.nli_model), acc: v.verification.accuracy, f1: v.verification.macro_f1_gold_classes, v }));
    if (judgeRuns.length > 1) {
      const order = ["General judge (bart-large)", "Small general judge (MiniLM)", "Science-trained judge (MiniLM)"];
      judgeRuns.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
      const series = [
        { name: "Accuracy", color: "var(--ink)", values: judgeRuns.map((j) => j.acc) },
        { name: "Macro-F1 (gold classes)", color: "var(--accent)", values: judgeRuns.map((j) => j.f1) },
      ];
      const ft = d.finetune_nli;
      const sci = judgeRuns.find((j) => /Science/.test(j.name)), small = judgeRuns.find((j) => /Small general/.test(j.name)), big = judgeRuns.find((j) => /General judge/.test(j.name));
      panels.push(`<div class="panel wide"><h2>Training the judge on science</h2>
        <p class="sub">Same search, same rule, different judge. Verdict accuracy on the 300 dev claims; the science-trained judge never saw them.</p>
        ${C.groupedBars({ categories: judgeRuns.map((j) => j.name.replace(/ \(.*\)/, "")), series, height: 230 })}
        <div class="legend">${series.map((s) => `<span style="--c:${s.color}">${s.name}</span>`).join("")}</div>
        <p class="note">${sci && small ? `Fine-tuning moved the small model from ${pct(small.acc, 1)} to ${pct(sci.acc, 1)}` : ""}${sci && big ? `; the ${sci.acc >= big.acc ? "82M-parameter science-trained judge now matches or beats" : "407M general judge still beats"} the ${sci.acc >= big.acc ? "407M general judge (" + pct(big.acc, 1) + ")" : "82M science-trained one (" + pct(big.acc, 1) + " vs " + pct(sci.acc, 1) + ")"}` : ""}.
        ${ft ? ` Trained on ${ft.examples.train.toLocaleString()} claim–paper pairs from ${ft.training_claims} training claims in ${Math.round(ft.training_time_seconds / 60)} min on a laptop CPU; best validation macro-F1 ${ft.best_val_macro_f1.toFixed(2)} (validation = a held-out 10% of train).` : ""}</p></div>`);
    }

    const calibrations = Object.entries(d).filter(([k]) => k.startsWith("calibration_")).map(([, v]) => v);
    calibrations.forEach((cal) => {
      const cl = cal.claim_level_dev, pl = cal.paper_level_dev;
      panels.push(`<div class="panel"><h2>Is the confidence honest? ${esc(judgeName(cal.nli_model).replace(/ \(.*\)/, ""))}</h2>
        <p class="sub">Each dot: verdicts that claimed a certain confidence, and how many were right. Dots on the dashed line are honest.</p>
        ${C.reliability({ series: [
          { name: "Before (raw)", color: "var(--CONTRADICTED)", bins: cl.before.bins },
          { name: `After (T = ${cal.temperature})`, color: "var(--SUPPORTED)", bins: cl.after.bins },
        ] })}
        <div class="legend"><span style="--c:var(--CONTRADICTED)">raw · ECE ${cl.before.ece.toFixed(3)}</span><span style="--c:var(--SUPPORTED)">calibrated · ECE ${cl.after.ece.toFixed(3)}</span></div>
        <p class="note">Decided verdicts on dev said on average ${pct(cl.before.mean_confidence)} sure but were right ${pct(cl.accuracy_of_decided)} of the time. One temperature (${cal.temperature}) fitted on ${cal.fitted_on.pairs} training pairs brings the stated confidence to ${pct(cl.after.mean_confidence)}. Paper-level ECE ${pl.before.ece.toFixed(3)} → ${pl.after.ece.toFixed(3)} over ${pl.pairs} pairs. No verdict changed.</p></div>`);
    });

    const ccs = Object.entries(d).filter(([k]) => k.startsWith("counterclaim_eval_") && !/_limit\d+$/.test(k)).map(([, v]) => v);
    ccs.forEach((cc) => {
      const st = cc.by_status, names = { consistent: "Consistent (opposite answers)", inconsistent: "Unreliable (same answer for both)", inconclusive: "Inconclusive (an UNKNOWN / CONFLICTING side)" };
      const items = ["consistent", "inconsistent", "inconclusive"].filter((k) => st[k]).map((k) => ({
        label: `${names[k]} · ${st[k].claims}`, value: st[k].error_rate, color: k === "inconsistent" ? "var(--CONTRADICTED)" : k === "consistent" ? "var(--SUPPORTED)" : "var(--NEUTRAL)" }));
      const dec = cc.decided_claims_only;
      panels.push(`<div class="panel"><h2>Does the opposite-claim check catch errors? ${esc(judgeName(cc.nli_model).replace(/ \(.*\)/, ""))}</h2>
        <p class="sub">Every dev claim and its opposite were both verified. Bars: how often the claim's verdict was wrong, by what the pair looked like.</p>
        ${C.hbars({ items, max: 1, labelWidth: 290 })}
        <p class="note">${st.inconsistent && st.consistent ? `Flagged verdicts are wrong ${pct(st.inconsistent.error_rate)} of the time against ${pct(st.consistent.error_rate)} for consistent ones. ` : ""}${dec && dec.accuracy_if_abstain_on_inconsistent != null ? `Among decided verdicts, abstaining on flagged ones would raise accuracy from ${pct(dec.accuracy, 1)} to ${pct(dec.accuracy_if_abstain_on_inconsistent, 1)} (keeping ${dec.kept} of ${dec.claims}).` : ""} Opposites were built by: ${Object.entries(cc.by_method).map(([m, n]) => `${m} ${n}`).join(", ")}.</p></div>`);
    });

    const systems = Object.entries(d)
      .filter(([k, v]) => k.startsWith("proposed_") && v.verification && v.num_claims === 300 && v.split !== "train" && /bart-large-mnli$/.test(v.nli_model || ""))
      .map(([k, v]) => ({
        label: `${v.retrieval.toUpperCase()} · ${v.sentence_selection ? `top ${v.sentences_per_doc || 2} sentences` : "whole abstract"}`,
        value: v.verification.accuracy, f1: v.verification.macro_f1_gold_classes,
        highlight: v.retrieval === "bm25" && !v.sentence_selection,
      }))
      .sort((a, b) => b.value - a.value);
    if (systems.length) {
      panels.push(`<div class="panel"><h2>What each idea changed</h2>
        <p class="sub">Verdict accuracy on 300 dev claims. Highlighted bar: the system this app runs.</p>
        ${C.hbars({ items: systems, max: 1, labelWidth: 220 })}
        <p class="note">None of the three improvements (hybrid search, sentence selection, more sentences) beat the plain version.</p></div>`);
    }

    if (ea && ea.confusion_matrix) {
      const order = ["SUPPORTED", "CONTRADICTED", "CONFLICTING", "UNKNOWN"];
      const cm = ea.confusion_matrix;
      const rows = order.filter((g) => cm[g]);
      let html = `<div class="confusion" style="grid-template-columns:110px repeat(${order.length},1fr)"><div></div>`;
      html += order.map((p) => `<div class="hd">${LABELS[p]}</div>`).join("");
      rows.forEach((g) => {
        const total = order.reduce((s, p) => s + (cm[g][p] || 0), 0) || 1;
        html += `<div class="hd rh">${LABELS[g]}</div>`;
        html += order.map((p) => {
          const n = cm[g][p] || 0, share = n / total;
          const col = g === p ? "var(--SUPPORTED)" : "var(--CONTRADICTED)";
          return `<div class="cell" title="Answer ${LABELS[g]}, system said ${LABELS[p]}: ${n} (${pct(share)})"
            style="background:color-mix(in srgb, ${col} ${Math.round(share * 85)}%, var(--panel-2));color:${share > .45 ? "#fff" : "var(--ink)"}">${n}</div>`;
        }).join("");
      });
      html += "</div>";
      panels.push(`<div class="panel"><h2>Where verdicts land</h2>
        <p class="sub">Rows: the dataset's answer. Columns: what the system said. Green diagonal is correct.</p>
        ${html}<p class="note">The biggest leak: 34 no-evidence claims called “contradicted”.</p></div>`);
    }

    if (ea && ea.by_category) {
      const names = {
        verdict_without_gold_evidence: "Verdict where the answer is “no evidence”",
        evidence_retrieved_not_recognised: "Right paper found, judge missed it",
        polarity_flip: "Support and contradict swapped",
        spurious_conflict: "False “conflicting”",
        retrieval_miss: "Right paper not found",
      };
      const items = Object.entries(ea.by_category).map(([k, v]) => ({ label: names[k] || k, value: v, color: k === "retrieval_miss" ? "var(--accent)" : "var(--CONTRADICTED)" }))
        .sort((a, b) => b.value - a.value);
      panels.push(`<div class="panel"><h2>Why it gets things wrong</h2>
        <p class="sub">${ea.errors} errors out of ${ea.claims} dev claims, by cause.</p>
        ${C.hbars({ items, format: (v) => String(Math.round(v)), labelWidth: 270 })}
        <p class="note">Only ${ea.by_category.retrieval_miss} errors come from search. The judging model is the weak part.</p></div>`);
    }

    if (ea && ea.judge_on_gold_documents) {
      const j = ea.judge_on_gold_documents;
      const items = [
        { label: "Claims with annotated evidence", value: j.gold_evidence_claims, highlight: true },
        { label: "Key paper retrieved", value: j.gold_doc_retrieved, highlight: true },
        { label: "Judge read it right", value: j.judge_right, highlight: true },
        { label: "Final verdict right", value: j.judge_right_final_right, color: "var(--SUPPORTED)" },
        { label: "Judge said neutral", value: j.judge_said_neutral, color: "var(--NEUTRAL)" },
        { label: "Judge said the opposite", value: j.judge_said_opposite, color: "var(--CONTRADICTED)" },
      ];
      panels.push(`<div class="panel"><h2>Following the key paper</h2>
        <p class="sub">What happens to the annotated evidence paper on its way to a verdict.</p>
        ${C.hbars({ items, format: (v) => String(Math.round(v)), labelWidth: 230 })}</div>`);
    }

    const sweep = d.threshold_sweep;
    if (sweep && sweep.train) {
      const xs = Object.keys(sweep.train).map(Number).sort((a, b) => a - b);
      const val = (split, x) => (sweep[split] && sweep[split][String(x)] ? sweep[split][String(x)].accuracy : null);
      const series = [{ name: "Train accuracy", color: "var(--ink)", values: xs.map((x) => val("train", x)) }];
      if (sweep.dev) series.push({ name: "Dev accuracy", color: "var(--accent)", values: xs.map((x) => val("dev", x)) });
      const all = series.flatMap((s) => s.values).filter((v) => v != null);
      panels.push(`<div class="panel"><h2>Tuning the evidence threshold</h2>
        <p class="sub">How sure the judge must be before a paper counts. Chosen on train, checked on dev.</p>
        ${C.line({ xs, series, yMin: Math.floor(Math.min(...all) * 20) / 20, yMax: Math.ceil(Math.max(...all) * 20) / 20, xFormat: (v) => v.toFixed(2), marker: { x: sweep.chosen_threshold_on_train, label: "chosen on train" } })}
        <div class="legend">${series.map((s) => `<span style="--c:${s.color}">${s.name}</span>`).join("")}</div>
        <p class="note">Try other thresholds yourself in Verify → Settings.</p></div>`);
    }

    const sel = d.sentence_selection_eval;
    if (sel) {
      const xs = Object.keys(sel).filter((k) => /^\d+$/.test(k)).map(Number).sort((a, b) => a - b);
      const series = [
        { name: "Best-matching sentences", color: "var(--SUPPORTED)", values: xs.map((x) => sel[x].selected_hit_rate) },
        { name: "First sentences", color: "var(--rule-strong)", values: xs.map((x) => sel[x].first_n_hit_rate) },
      ];
      panels.push(`<div class="panel"><h2>Picking the evidence sentences</h2>
        <p class="sub">How often the chosen sentences include an annotated rationale sentence (${sel[xs[0]].pairs} claim–paper pairs).</p>
        ${C.line({ xs, series, xFormat: (v) => `${v} sent.` })}
        <div class="legend">${series.map((s) => `<span style="--c:${s.color}">${s.name}</span>`).join("")}</div>
        <p class="note">This is how each evidence card picks its highlighted sentences.</p></div>`);
    }

    return `<div class="kpis">${kpis.join("")}</div><div class="bench-grid">${panels.join("")}</div>
      <p class="fineprint">Model: ${esc(base ? base.nli_model : "")}. Accuracy means the share of claims where the verdict equals the dataset label. The confidence values shown in Verify are not calibrated.</p>`;
  }

  // ------------------------------------------------------------------ keyboard
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) || reader.open;
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") { e.preventDefault(); go("verify"); claimEl.focus(); }
    else if (e.key === "t") toggleTheme();
    else if (e.key === "r" && parseHash().route === "verify") randomClaim();
  });

  // ------------------------------------------------------------------ start
  setTopK(state.topK);
  setThr(state.thr);
  renderExamples();
  renderHistory();
  updateHints();
  onRoute();
  $("#judge").closest(".judge-pick").hidden = true;
  loadJudges().finally(() => {
    api(`/warmup${state.judge ? "?model=" + encodeURIComponent(state.judge) : ""}`, { method: "POST" }).catch(() => {}).finally(pollStatus);
  });
})();
