// app.js -- interface for the catalyst-requirement pipeline. Uses window.Engine (engine.js).
(function () {
  "use strict";
  const E = window.Engine, $ = id => document.getElementById(id);
  const PRESETS = E.presets();
  const state = { last: null };

  // ---------- inputs ----------
  function readInputs() {
    const molecule = $("molecule").value;
    let spec;
    if (molecule === "chain") spec = E.chain(+$("ld").value, +$("le").value, $("endo").checked);
    else spec = PRESETS[molecule];
    const fe = document.querySelector('input[name="fe"]:checked').value;
    return {
      molecule, spec, gamma: +$("gamma").value, hO: +$("ho").value, kset: $("kset").value, fe,
      analog: { adcBits: +$("adc").value, fitted: $("range").value === "fitted", gain: +$("gain").value / 100,
                offset: +$("offset").value, noise: +$("noise").value, seed: +$("chip").value, wBits: 8, dBits: 8 },
      rseed: +$("rseed").value,
    };
  }
  function syncLabels() {
    $("gamma-v").textContent = (+$("gamma").value).toFixed(2);
    $("ho-v").textContent = (+$("ho").value).toFixed(2);
    $("ld-v").textContent = $("ld").value; $("le-v").textContent = $("le").value;
    $("adc-v").textContent = $("adc").value + " bits";
    $("gain-v").textContent = (+$("gain").value).toFixed(2) + "%";
    $("offset-v").textContent = (+$("offset").value).toExponential(0).replace("0e+0", "0");
    $("noise-v").textContent = (+$("noise").value).toExponential(0).replace("0e+0", "0");
    $("chain-opts").hidden = $("molecule").value !== "chain";
    $("analog-opts").hidden = document.querySelector('input[name="fe"]:checked').value !== "analog";
    const spec = $("molecule").value === "chain" ? E.chain(+$("ld").value, +$("le").value, $("endo").checked) : PRESETS[$("molecule").value];
    $("ho-row").hidden = !spec.atoms.some(a => a.t === "O=");
  }

  // ---------- run ----------
  function run() {
    syncLabels();
    const inp = readInputs(), spec = inp.spec, p = { hO: inp.hO, kset: inp.kset };
    const t0 = performance.now();
    const { H, no } = E.buildH(spec, inp.gamma, p), N = H.length;
    const ex = E.exact(H, no, spec.form);
    const fe = E.frontEnd(inp.fe, inp.analog, N);
    const sp = E.sp2(H, no, fe);
    const rf = E.refine(sp.X, H, no, spec.form, ex.map, { seed: inp.rseed, kmax: 10, target: 1e-5 });
    const ok = rf.reachedAt !== null;
    const refined = ok ? rf.last : null;
    const rows = ok ? E.ruleTable(spec, inp.gamma, p, refined) : null;
    const ms = performance.now() - t0;
    const gap = ex.e[no] - ex.e[no - 1];
    const I = ok ? imbalance(refined.Co, spec.form) : null;
    state.last = { inp, spec, H, no, ex, sp, rf, ok, rows, ms, gap, I };
    render();
  }
  function imbalance(Co, form) {
    const [[a, b], [c, d]] = form; let s = 0;
    for (let i = 0; i < Co[0].length; i++) s += 2 * (Co[a][i] * Co[b][i] - Co[c][i] * Co[d][i]);
    return s;
  }

  // ---------- rendering ----------
  const FE_NAME = { fp64: "float64 (exact digital)", fp32: "float32", fp16pair: "FP16-pair, FP32 accumulate", int8: "INT8 digital", analog: "Analog array (behavioral model)" };
  function render() {
    const L = state.last;
    renderSummary(L); renderMolecule(L); renderMatrix(L); renderSP2(L); renderRefine(L); renderMap(L); renderTable(L); renderReport(L);
  }
  function fmt(v, d) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d ?? 4); }
  function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

  function renderSummary(L) {
    const pill = $("status");
    if (L.ok) { pill.className = "pill good"; pill.textContent = `Refined to 1e-5 in ${L.rf.reachedAt} Ogita–Aishima step${L.rf.reachedAt === 1 ? "" : "s"}`; }
    else { pill.className = "pill bad"; pill.textContent = L.sp.diverged ? "Front end diverged" : "Refinement did not converge"; }
    $("meta").textContent = `${L.spec.atoms.length} π centres · ${L.no} occupied orbitals · HOMO–LUMO gap ${L.gap.toFixed(3)}|β| · ${FE_NAME[L.inp.fe]} · ${L.ms.toFixed(0)} ms`;
    const box = $("reqs");
    if (!L.ok) {
      box.innerHTML = `<p class="warn">The front end did not land the start inside the refiner’s convergence basin, so no catalyst profile is reported. Raise the ADC resolution, lower the noise, or pick a more precise front end. This is the precision budget at work: larger molecules need more bits.</p>`;
      return;
    }
    const sel = E.summary(L.rows, 2);
    if (!sel.length) { box.innerHTML = `<p class="muted">No electrostatic lever at this level of theory. In an alternant hydrocarbon pair every charge stays at 1 and the forming bonds stay balanced, so the Hückel model gives a catalyst nothing to act on.</p>`; return; }
    box.innerHTML = sel.map(r => `<li class="req ${r.sign > 0 ? "pos" : "neg"}">
        <span class="req-site">${esc(r.name)}<small>${r.frag}</small></span>
        <span class="req-what">${esc(r.env.split(" (")[0])} environment<small>${esc(r.env.includes("(") ? r.env.slice(r.env.indexOf("(") + 1, -1) : "")}</small></span>
        <span class="req-num">${fmt(r.LA)}<small>${r.reach} · sign stable in ${(100 * r.robust).toFixed(0)}% of variants</small></span></li>`).join("");
  }

  // Molecule: s-cis diene arc on top, dienophile underneath, forming bonds dashed; halo = lever.
  function layout(spec) {
    const s = 52, pos = new Array(spec.atoms.length);
    const dC = spec.atoms.map((a, i) => i).filter(i => spec.atoms[i].frag === "diene" && spec.atoms[i].t === "C");
    const dp = spec.atoms.map((a, i) => i).filter(i => spec.atoms[i].frag === "dienophile");
    const x0 = 60, y1 = 120, y2 = 210;
    const U = [[x0, y1], [x0 + 0.5 * s, y1 - 0.87 * s], [x0 + 1.5 * s, y1 - 0.87 * s], [x0 + 2 * s, y1]];
    dC.forEach((idx, j) => { if (j < 4) pos[idx] = U[j]; else { const prev = pos[dC[j - 1]]; pos[idx] = [prev[0] + 0.87 * s, j % 2 === 0 ? y1 : y1 - 0.5 * s]; } });
    dp.forEach((idx, j) => {
      if (j === 0) pos[idx] = [x0, y2]; else if (j === 1) pos[idx] = [x0 + 2 * s, y2];
      else { const prev = pos[dp[j - 1]]; pos[idx] = [prev[0] + 0.87 * s, j % 2 === 0 ? y2 : y2 + 0.5 * s]; } });
    spec.atoms.forEach((a, i) => { if (!pos[i]) { // substituent: above its partner
      const b = spec.bonds.find(([u, v]) => u === i || v === i), o = b[0] === i ? b[1] : b[0];
      pos[i] = [pos[o][0], pos[o][1] - 0.9 * s]; } });
    return pos;
  }
  function renderMolecule(L) {
    const spec = L.spec, pos = layout(spec), lev = L.ok ? L.rows.map(r => r.LA) : null;
    const mx = lev ? Math.max(1e-12, ...lev.map(Math.abs)) : 1;
    const xs = pos.map(p => p[0]), ys = pos.map(p => p[1]);
    const minX = Math.min(...xs) - 50, maxX = Math.max(...xs) + 50, minY = Math.min(...ys) - 50, maxY = Math.max(...ys) + 55;
    let g = "";
    spec.bonds.forEach(([i, j, kind]) => {
      const [a, b] = [pos[i], pos[j]];
      g += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" class="bond"/>`;
      if (kind.includes("=")) { const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy), ox = -dy / l * 4, oy = dx / l * 4;
        g += `<line x1="${a[0] + ox}" y1="${a[1] + oy}" x2="${b[0] + ox}" y2="${b[1] + oy}" class="bond thin"/>`; } });
    spec.form.forEach(([i, j]) => { const [a, b] = [pos[i], pos[j]];
      g += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" class="forming"/>`; });
    spec.atoms.forEach((at, i) => {
      const [x, y] = pos[i];
      if (lev && Math.abs(lev[i]) > 1e-9) { const r = 14 + 22 * Math.abs(lev[i]) / mx;
        g += `<circle cx="${x}" cy="${y}" r="${r.toFixed(1)}" class="halo ${lev[i] > 0 ? "pos" : "neg"}"><title>${esc(at.name)}: lever ${fmt(lev[i])}</title></circle>`; }
      g += `<circle cx="${x}" cy="${y}" r="13" class="atom ${at.t === "C" ? "" : "hetero"}"/>`;
      g += `<text x="${x}" y="${y + 4}" class="atom-label">${esc(at.name.replace("(Me)", ""))}</text>`;
    });
    $("mol").innerHTML = `<svg viewBox="${minX} ${minY} ${maxX - minX} ${maxY - minY}" style="max-width:${Math.round(1.25 * (maxX - minX))}px;margin:0 auto" role="img" aria-label="Diene above, dienophile below, forming bonds dashed">${g}</svg>`;
    $("mol-note").textContent = L.ok
      ? "Halo size = how strongly an environment at that atom shifts the forming-bond imbalance; teal wants an electron-withdrawing contact, vermilion an electron-donating one."
      : "Halos appear once the refinement converges.";
  }

  function renderMatrix(L) {
    const H = L.H, N = H.length, c = Math.max(8, Math.min(24, Math.floor(420 / N))), lab = N <= 16, pad = lab ? 34 : 6;
    let mx = 0; H.forEach(r => r.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
    const formSet = new Set(L.spec.form.flatMap(([i, j]) => [i + "," + j, j + "," + i]));
    let g = "";
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = H[i][j], a = Math.abs(v) / mx;
      g += `<rect x="${pad + j * c}" y="${pad + i * c}" width="${c - 1}" height="${c - 1}" class="${v === 0 ? "cell0" : (formSet.has(i + "," + j) ? "cellf" : "cell")}" style="fill-opacity:${v === 0 ? 1 : (0.15 + 0.85 * a).toFixed(2)}"><title>H[${L.spec.atoms[i].name},${L.spec.atoms[j].name}] = ${v.toFixed(3)}</title></rect>`;
    }
    if (lab) L.spec.atoms.forEach((a, i) => {
      g += `<text x="${pad - 5}" y="${pad + i * c + c / 2 + 3}" class="axis" text-anchor="end">${esc(a.name)}</text>`;
      g += `<text x="${pad + i * c + c / 2}" y="${pad - 6}" class="axis" text-anchor="middle">${esc(a.name)}</text>`; });
    const W = pad + N * c + 4;
    $("matrix").innerHTML = `<svg viewBox="0 0 ${W} ${W}" style="max-width:${W}px" role="img" aria-label="Hückel matrix heatmap">${g}</svg>`;
    $("matrix-note").textContent = `${N}×${N}, units of |β|. Darker = larger magnitude; the two outlined amber cells are the forming bonds at γ = ${L.inp.gamma.toFixed(2)}.`;
  }

  // Log-scale line chart drawn by hand.
  function logChart(el, ys, o) {
    const W = 520, Hh = 210, l = 52, r = 14, t = 14, b = 34;
    const vals = ys.filter(v => Number.isFinite(v) && v > 0);
    const lo = Math.floor(Math.log10(Math.min(o.floor ?? 1e-16, ...vals, o.ref ?? 1))), hi = Math.ceil(Math.log10(Math.max(...vals, o.ref ?? 1e-16, 1)));
    const n = Math.max(ys.length - 1, 1), X = i => l + (W - l - r) * i / n, Y = v => t + (Hh - t - b) * (hi - Math.log10(v)) / (hi - lo || 1);
    let g = "";
    for (let d = lo; d <= hi; d += Math.max(1, Math.round((hi - lo) / 6))) {
      g += `<line x1="${l}" x2="${W - r}" y1="${Y(10 ** d)}" y2="${Y(10 ** d)}" class="grid"/><text x="${l - 6}" y="${Y(10 ** d) + 4}" class="axis" text-anchor="end">1e${d}</text>`; }
    ys.forEach((_, i) => { if (n <= 12 || i % Math.ceil(n / 10) === 0) g += `<text x="${X(i)}" y="${Hh - b + 16}" class="axis" text-anchor="middle">${i + (o.x0 ?? 0)}</text>`; });
    g += `<text x="${(l + W - r) / 2}" y="${Hh - 4}" class="axis" text-anchor="middle">${o.xlabel}</text>`;
    if (o.ref) g += `<line x1="${l}" x2="${W - r}" y1="${Y(o.ref)}" y2="${Y(o.ref)}" class="ref"/><text x="${W - r}" y="${Y(o.ref) - 6}" class="axis" text-anchor="end">${o.refLabel}</text>`;
    const pts = ys.map((v, i) => Number.isFinite(v) && v > 0 ? `${X(i)},${Y(Math.max(v, 10 ** lo))}` : null).filter(Boolean);
    g += `<polyline points="${pts.join(" ")}" class="line"/>`;
    ys.forEach((v, i) => { if (Number.isFinite(v) && v > 0) g += `<circle cx="${X(i)}" cy="${Y(Math.max(v, 10 ** lo))}" r="${i === o.mark ? 5.5 : 3}" class="${i === o.mark ? "dot-mark" : "dot"}"><title>${o.xlabel} ${i + (o.x0 ?? 0)}: ${v.toExponential(2)}</title></circle>`;
      else g += `<text x="${X(i)}" y="${t + 10}" class="axis bad-t" text-anchor="middle">∞</text>`; });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${o.aria}">${g}</svg>`;
  }
  function renderSP2(L) {
    logChart($("sp2"), L.sp.hist, { xlabel: "SP2 iteration", x0: 1, aria: "SP2 idempotency error per iteration", floor: 1e-16 });
    $("sp2-note").textContent = `${L.sp.iters} matrix-product iterations in ${FE_NAME[L.inp.fe]}${L.sp.diverged ? "; diverged" : ""}. The curve flattens at the format’s noise floor; that floor, not the curve’s end, is what the refiner inherits.`;
  }
  function renderRefine(L) {
    const ys = L.rf.steps.map(s => s.err);
    logChart($("refine"), ys, { xlabel: "Ogita–Aishima step", x0: 0, ref: 1e-5, refLabel: "target 1e-5", mark: L.rf.reachedAt ?? -1, aria: "Sensitivity-map error per refinement step", floor: 1e-16 });
    $("refine-note").textContent = L.ok
      ? `Float64 refinement reached the target at step ${L.rf.reachedAt}. Start: range finder plus one Rayleigh–Ritz per block; each step is four dense matrix products.`
      : "The start was outside the basin, so the refinement drifted. A cold start fails the same way; the front end’s job is to land the start inside the basin.";
  }
  function renderMap(L) {
    if (!L.ok) { $("map").innerHTML = `<p class="muted">No map: the refinement did not converge.</p>`; return; }
    const rows = L.rows, mx = Math.max(1e-12, ...rows.map(r => Math.abs(r.LA)), ...rows.map(r => Math.abs(r.LE)));
    const W = 520, rowH = 20, l = 70, mid = l + (W - l - 20) / 2, half = (W - l - 20) / 2, Hh = rows.length * rowH + 40;
    let g = `<line x1="${mid}" x2="${mid}" y1="8" y2="${Hh - 28}" class="axisline"/>`;
    rows.forEach((r, i) => {
      const y = 12 + i * rowH, w = half * Math.abs(r.LA) / mx, x = r.LA >= 0 ? mid : mid - w;
      g += `<text x="${l - 8}" y="${y + 11}" class="axis" text-anchor="end">${esc(r.name)}</text>`;
      g += `<rect x="${x}" y="${y}" width="${Math.max(w, 0.5)}" height="13" rx="2" class="bar ${r.LA >= 0 ? "pos" : "neg"}"><title>${esc(r.name)}: asynchronicity lever ${fmt(r.LA)}, interaction lever ${fmt(r.LE)}</title></rect>`;
      const xe = mid + half * r.LE / mx;
      g += `<line x1="${xe}" x2="${xe}" y1="${y - 1}" y2="${y + 14}" class="tick-e"><title>interaction lever ${fmt(r.LE)}</title></line>`;
    });
    g += `<text x="${mid - 6}" y="${Hh - 10}" class="axis" text-anchor="end">← wants donating</text><text x="${mid + 6}" y="${Hh - 10}" class="axis">wants withdrawing →</text>`;
    $("map").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Per-atom levers">${g}</svg>`;
    $("map-note").textContent = `Bars: asynchronicity lever dI/dδ (primary). Thin marks: interaction lever q(γ) − q(0). Current imbalance I = ${fmt(L.I)}; scale ±${mx.toFixed(4)}.`;
  }
  function renderTable(L) {
    if (!L.ok) { $("table").innerHTML = ""; return; }
    const rows = L.rows.slice().sort((a, b) => a.frag === b.frag ? a.rank - b.rank : (a.frag === "diene" ? -1 : 1));
    $("table").innerHTML = `<table><thead><tr><th>Atom</th><th>Fragment</th><th class="num">Async. lever</th><th class="num">Interaction lever</th><th>Requirement</th><th>Reach</th><th class="num">Rank</th><th class="num">Sign stable</th></tr></thead><tbody>${
      rows.map(r => `<tr><td><span class="sw ${r.sign > 0 ? "pos" : r.sign < 0 ? "neg" : ""}"></span>${esc(r.name)}</td><td>${r.frag}</td><td class="num">${fmt(r.LA)}</td><td class="num">${fmt(r.LE)}</td><td>${esc(r.env)}</td><td>${r.reach}</td><td class="num">${r.rank}</td><td class="num">${(100 * r.robust).toFixed(0)}%</td></tr>`).join("")
    }</tbody></table>`;
  }
  function reportText(L) {
    const i = L.inp, a = i.analog;
    const fe = i.fe === "analog" ? `${FE_NAME.analog}: 8-bit weights, 8-bit duty, ${a.adcBits}-bit ADC (${a.fitted ? "fitted" : "worst-case"} range), gain σ ${(100 * a.gain).toFixed(2)}%, offset ${a.offset}, noise ${a.noise} of full scale, chip ${a.seed}` : FE_NAME[i.fe];
    const molName = i.molecule === "chain" ? `extended chain (diene ${$("ld").value} atoms, dienophile ${$("le").value} atoms${$("endo").checked ? ", ends in C=O" : ""})` : i.molecule;
    let s = `# Catalyst requirement report\n\nPair: ${molName}\nCoupling at forming bonds: γ = ${i.gamma.toFixed(2)}|β|; h_O = ${i.hO.toFixed(2)}; bond set: ${i.kset}\n`;
    s += `Size: ${L.spec.atoms.length} π centres, ${L.no} occupied orbitals, HOMO–LUMO gap ${L.gap.toFixed(3)}|β|\n`;
    s += `Front end: ${fe}; SP2 iterations: ${L.sp.iters}${L.sp.diverged ? " (diverged)" : ""}\n`;
    s += `Refinement: ${L.ok ? `reached map error ≤ 1e-5 at Ogita–Aishima step ${L.rf.reachedAt}` : "did not converge (start outside the basin)"}\n\n`;
    if (!L.ok) return s + "No catalyst profile: the front end was not precise enough for this molecule.\n";
    const sel = E.summary(L.rows, 2);
    s += `## Ideal catalyst profile\n\n`;
    if (!sel.length) s += "No electrostatic lever at this level of theory.\n";
    sel.forEach((r, n) => { s += `${n + 1}. ${r.frag}, ${r.name}: ${r.env} [${r.reach}]; lever ${fmt(r.LA)}; sign stable in ${(100 * r.robust).toFixed(0)}% of parameter variants${r.alsoBind ? "; also strengthens the reactant interaction" : ""}\n`; });
    s += `\n## Per-atom levers\n\n| Atom | Fragment | Async. lever | Interaction lever | Requirement | Sign stable |\n|---|---|---|---|---|---|\n`;
    L.rows.forEach(r => { s += `| ${r.name} | ${r.frag} | ${fmt(r.LA)} | ${fmt(r.LE)} | ${r.env} | ${(100 * r.robust).toFixed(0)}% |\n`; });
    s += `\nMethod: simple Hückel supermolecule; asynchronicity lever dI/dδ (primary, after DFT work linking Lewis-acid catalysis to asynchronicity and reduced Pauli repulsion); interaction lever q(γ) − q(0). Heuristic only: no σ framework, sterics, solvent, geometry or Pauli repulsion.\n`;
    return s;
  }
  function renderReport(L) { $("report").value = reportText(L); }
  function reportJSON(L) {
    return JSON.stringify({ inputs: L.inp, converged: L.ok, reachedAt: L.rf.reachedAt, sp2Iterations: L.sp.iters, gap: L.gap,
      imbalance: L.I, rows: L.rows }, (k, v) => (k === "spec" ? undefined : (typeof v === "number" ? +v.toPrecision(6) : v)), 2);
  }
  async function copy(text, btn) {
    const orig = btn.textContent;
    try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
    catch (e) { const ta = $("report"); ta.value = text; ta.focus(); ta.select(); btn.textContent = "Selected: press Ctrl+C"; }
    setTimeout(() => { btn.textContent = orig; }, 1800);
  }

  // ---------- wiring ----------
  let timer = null;
  function schedule() { syncLabels(); clearTimeout(timer); timer = setTimeout(run, 60); }
  document.querySelectorAll("#controls input, #controls select").forEach(el => el.addEventListener("input", schedule));
  $("copy-md").addEventListener("click", e => copy(reportText(state.last), e.currentTarget));
  $("copy-json").addEventListener("click", e => copy(reportJSON(state.last), e.currentTarget));
  run();
})();
