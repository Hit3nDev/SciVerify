/* Tiny dependency-free SVG charts for the benchmarks page. Colours are CSS variables, so the
   charts follow the light/dark theme. Every function returns an SVG string. */
(function () {
  "use strict";

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (v) => (v * 100).toFixed(1) + "%";

  /* Vertical grouped bars: categories along x, one bar per series in each group. */
  function groupedBars({ categories, series, max = 1, format = pct, height = 240 }) {
    const W = 560, H = height, L = 40, R = 8, T = 22, B = 30;
    const plotW = W - L - R, plotH = H - T - B;
    const groupW = plotW / categories.length;
    const barW = Math.min(46, (groupW * 0.78) / series.length);
    const y = (v) => T + plotH - (v / max) * plotH;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">`;
    for (let i = 0; i <= 4; i++) {
      const v = (max * i) / 4, yy = y(v);
      s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>`;
      s += `<text x="${L - 6}" y="${yy + 4}" text-anchor="end">${esc(format(v).replace(".0%", "%"))}</text>`;
    }
    categories.forEach((cat, ci) => {
      const gx = L + ci * groupW + (groupW - barW * series.length) / 2;
      series.forEach((ser, si) => {
        const v = ser.values[ci];
        if (v == null) return;
        const x = gx + si * barW, yy = y(v), h = T + plotH - yy;
        s += `<rect x="${x + 2}" y="${yy}" width="${barW - 4}" height="${Math.max(h, 0)}" rx="2" style="fill:${ser.color}"><title>${esc(ser.name)} · ${esc(cat)}: ${esc(format(v))}</title></rect>`;
        s += `<text class="val" x="${x + barW / 2}" y="${yy - 5}" text-anchor="middle">${esc(format(v))}</text>`;
      });
      s += `<text x="${L + ci * groupW + groupW / 2}" y="${H - 10}" text-anchor="middle">${esc(cat)}</text>`;
    });
    s += `<line class="axis" x1="${L}" x2="${W - R}" y1="${T + plotH}" y2="${T + plotH}"/></svg>`;
    return s;
  }

  /* Horizontal bars, one per item; `highlight` items are drawn in ink, others muted. */
  function hbars({ items, max, format = pct, labelWidth = 230 }) {
    const rowH = 30, W = 560, L = labelWidth, R = 56, T = 4;
    const H = T + items.length * rowH + 4;
    const m = max || Math.max(...items.map((d) => d.value)) || 1;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">`;
    items.forEach((d, i) => {
      const yy = T + i * rowH, w = ((W - L - R) * d.value) / m;
      s += `<text x="${L - 10}" y="${yy + rowH / 2 + 4}" text-anchor="end"${d.highlight ? ' style="fill:var(--ink);font-weight:600"' : ""}>${esc(d.label)}</text>`;
      s += `<rect x="${L}" y="${yy + 6}" width="${W - L - R}" height="${rowH - 12}" rx="2" style="fill:var(--panel-2)"/>`;
      s += `<rect x="${L}" y="${yy + 6}" width="${Math.max(w, 1)}" height="${rowH - 12}" rx="2" style="fill:${d.color || (d.highlight ? "var(--ink)" : "var(--rule-strong)")}"><title>${esc(d.label)}: ${esc(format(d.value))}</title></rect>`;
      s += `<text class="val" x="${L + Math.max(w, 1) + 6}" y="${yy + rowH / 2 + 4}">${esc(format(d.value))}</text>`;
    });
    return s + "</svg>";
  }

  /* Line chart over numeric x values, with an optional vertical marker. */
  function line({ xs, series, yMin = 0, yMax = 1, format = pct, xFormat = String, marker = null, height = 240 }) {
    const W = 560, H = height, L = 44, R = 12, T = 16, B = 30;
    const plotW = W - L - R, plotH = H - T - B;
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const x = (v) => L + ((v - x0) / (x1 - x0 || 1)) * plotW;
    const y = (v) => T + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">`;
    for (let i = 0; i <= 4; i++) {
      const v = yMin + ((yMax - yMin) * i) / 4, yy = y(v);
      s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>`;
      s += `<text x="${L - 6}" y="${yy + 4}" text-anchor="end">${esc(format(v))}</text>`;
    }
    xs.forEach((v) => { s += `<text x="${x(v)}" y="${H - 10}" text-anchor="middle">${esc(xFormat(v))}</text>`; });
    if (marker != null) {
      s += `<line x1="${x(marker.x)}" x2="${x(marker.x)}" y1="${T}" y2="${T + plotH}" style="stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 4"/>`;
      s += `<text x="${x(marker.x) + 6}" y="${T + 10}" style="fill:var(--accent);font-weight:600">${esc(marker.label)}</text>`;
    }
    series.forEach((ser) => {
      const pts = xs.map((xv, i) => (ser.values[i] == null ? null : [x(xv), y(ser.values[i])])).filter(Boolean);
      s += `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" style="fill:none;stroke:${ser.color};stroke-width:2.2;stroke-linejoin:round"/>`;
      pts.forEach((p, i) => {
        s += `<circle cx="${p[0]}" cy="${p[1]}" r="3.4" style="fill:var(--panel);stroke:${ser.color};stroke-width:2"><title>${esc(ser.name)} @ ${esc(xFormat(xs[i]))}: ${esc(format(ser.values[i]))}</title></circle>`;
      });
    });
    s += `<line class="axis" x1="${L}" x2="${W - R}" y1="${T + plotH}" y2="${T + plotH}"/></svg>`;
    return s;
  }

  window.Charts = { groupedBars, hbars, line, pct };
})();
