// engine.js -- numerical core of the catalyst-requirement pipeline (browser + Node).
// Hückel model in units of |beta| (alpha_C = 0, beta < 0): H_ii = -h_i, H_ij = -k_ij, forming bonds -gamma.
// Mirrors the Python scripts (rules.py, scale.py, s2_oa.py, mc_budget.py); checked against them in test_engine.js.
(function (root) {
  "use strict";
  const E_PI = { "C": 1, "O=": 1, "OR": 2 };
  const KSETS = {
    std: { "C=C": 1.0, "C-C": 1.0, "C=O": 1.0, "C-OR": 0.8 },
    "2312": { "C=C": 1.1, "C-C": 0.9, "C=O": 1.0, "C-OR": 0.8 },
  };

  // ---------- molecules ----------
  function butadiene() {
    return {
      atoms: [{ t: "C", frag: "diene", name: "C1" }, { t: "C", frag: "diene", name: "C2" },
              { t: "C", frag: "diene", name: "C3" }, { t: "C", frag: "diene", name: "C4" }],
      bonds: [[0, 1, "C=C"], [1, 2, "C-C"], [2, 3, "C=C"]],
    };
  }
  function presets() {
    const out = {};
    let d = butadiene();
    out["butadiene + ethylene"] = {
      atoms: d.atoms.concat([{ t: "C", frag: "dienophile", name: "C5" }, { t: "C", frag: "dienophile", name: "C6" }]),
      bonds: d.bonds.concat([[4, 5, "C=C"]]), form: [[0, 4], [3, 5]],
    };
    d = butadiene();
    out["butadiene + acrolein"] = {
      atoms: d.atoms.concat([{ t: "C", frag: "dienophile", name: "Cb" }, { t: "C", frag: "dienophile", name: "Ca" },
                             { t: "C", frag: "dienophile", name: "Cc" }, { t: "O=", frag: "dienophile", name: "O" }]),
      bonds: d.bonds.concat([[4, 5, "C=C"], [5, 6, "C-C"], [6, 7, "C=O"]]), form: [[0, 4], [3, 5]],
    };
    d = butadiene();
    out["2-methoxybutadiene + acrolein"] = {
      atoms: d.atoms.concat([{ t: "OR", frag: "diene", name: "O(Me)" },
                             { t: "C", frag: "dienophile", name: "Cb" }, { t: "C", frag: "dienophile", name: "Ca" },
                             { t: "C", frag: "dienophile", name: "Cc" }, { t: "O=", frag: "dienophile", name: "O" }]),
      bonds: d.bonds.concat([[1, 4, "C-OR"], [5, 6, "C=C"], [6, 7, "C-C"], [7, 8, "C=O"]]), form: [[0, 5], [3, 6]],
    };
    return out;
  }
  // Extended family: polyene diene (LD atoms) + polyene / polyenal dienophile (LE atoms), forming bonds C1-Cb, C4-Ca.
  function chain(LD, LE, endO) {
    const atoms = [], bonds = [];
    for (let i = 0; i < LD; i++) atoms.push({ t: "C", frag: "diene", name: "C" + (i + 1) });
    for (let i = 0; i < LD - 1; i++) bonds.push([i, i + 1, i % 2 === 0 ? "C=C" : "C-C"]);
    const names = ["Cb", "Ca"];
    for (let i = 0; i < LE; i++) {
      const last = i === LE - 1;
      const t = endO && last ? "O=" : "C";
      const nm = endO && last ? "O" : (i < 2 ? names[i] : "D" + (i + 1));
      atoms.push({ t, frag: "dienophile", name: nm });
    }
    for (let i = 0; i < LE - 1; i++) {
      const last = i === LE - 2;
      bonds.push([LD + i, LD + i + 1, endO && last ? "C=O" : (i % 2 === 0 ? "C=C" : "C-C")]);
    }
    return { atoms, bonds, form: [[0, LD], [3, LD + 1]] };
  }

  function buildH(spec, gamma, p) {
    p = p || {};
    const hO = p.hO ?? 1.0, hOR = p.hOR ?? 2.0, ks = Object.assign({}, KSETS[p.kset || "std"], p.kOverride || {});
    const N = spec.atoms.length, H = zeros(N, N);
    spec.atoms.forEach((a, i) => { H[i][i] = -({ "C": 0, "O=": hO, "OR": hOR })[a.t]; });
    spec.bonds.forEach(([i, j, kind]) => { H[i][j] = H[j][i] = -ks[kind]; });
    spec.form.forEach(([i, j]) => { H[i][j] = H[j][i] = -gamma; });
    const ne = spec.atoms.reduce((s, a) => s + E_PI[a.t], 0);
    return { H, no: ne / 2 };
  }

  // ---------- linear algebra (dense, small N) ----------
  function zeros(n, m) { const A = new Array(n); for (let i = 0; i < n; i++) A[i] = new Array(m).fill(0); return A; }
  function eye(n) { const A = zeros(n, n); for (let i = 0; i < n; i++) A[i][i] = 1; return A; }
  function matmul(A, B) {
    const n = A.length, k = B.length, m = B[0].length, C = zeros(n, m);
    for (let i = 0; i < n; i++) { const Ai = A[i], Ci = C[i];
      for (let p = 0; p < k; p++) { const a = Ai[p]; if (a === 0) continue; const Bp = B[p];
        for (let j = 0; j < m; j++) Ci[j] += a * Bp[j]; } }
    return C;
  }
  function T(A) { const n = A.length, m = A[0].length, B = zeros(m, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) B[j][i] = A[i][j]; return B; }
  function trace(A) { let s = 0; for (let i = 0; i < A.length; i++) s += A[i][i]; return s; }
  function map2(A, f) { return A.map((r, i) => r.map((v, j) => f(v, i, j))); }
  function cols(A, idx) { return A.map(r => idx.map(j => r[j])); }
  function hcat(A, B) { return A.map((r, i) => r.concat(B[i])); }
  function maxAbs(A) { let m = 0; for (const r of A) for (const v of r) { const a = Math.abs(v); if (a > m) m = a; } return m; }
  function finite(A) { for (const r of A) for (const v of r) if (!Number.isFinite(v)) return false; return true; }

  // Cyclic Jacobi eigensolver for symmetric matrices; eigenvalues ascending, eigenvectors as columns.
  function eigh(A0) {
    const n = A0.length, A = A0.map(r => r.slice()), V = eye(n);
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0, tot = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const s = A[i][j] * A[i][j]; tot += s; if (i !== j) off += s; }
      if (off <= 1e-30 * Math.max(tot, 1e-300)) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = A[p][q]; if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    const order = [...Array(n).keys()].sort((a, b) => A[a][a] - A[b][b]);
    return { e: order.map(i => A[i][i]), V: V.map(r => order.map(i => r[i])) };
  }
  function qr(A) { // modified Gram-Schmidt, returns Q with orthonormal columns
    const n = A.length, m = A[0].length, Q = A.map(r => r.slice());
    for (let j = 0; j < m; j++) {
      for (let k = 0; k < j; k++) { let d = 0; for (let i = 0; i < n; i++) d += Q[i][k] * Q[i][j];
        for (let i = 0; i < n; i++) Q[i][j] -= d * Q[i][k]; }
      let nr = 0; for (let i = 0; i < n; i++) nr += Q[i][j] * Q[i][j]; nr = Math.sqrt(nr) || 1;
      for (let i = 0; i < n; i++) Q[i][j] /= nr;
    }
    return Q;
  }

  // ---------- seeded randomness ----------
  function rng(seed) { let a = (seed >>> 0) || 1;
    const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    const gauss = () => { if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
    return { u, gauss };
  }

  // ---------- number formats ----------
  const f32 = Math.fround;
  function half(x) { // round to nearest IEEE half (with subnormals); overflow -> Infinity
    if (x === 0 || !Number.isFinite(x)) return x;
    const ax = Math.abs(x); if (ax >= 65520) return Math.sign(x) * Infinity;
    let e = Math.floor(Math.log2(ax)); if (e < -14) e = -14;
    const sp = Math.pow(2, e - 10); return Math.sign(x) * Math.round(ax / sp) * sp;
  }
  function mmF32(A, B) { const n = A.length, k = B.length, m = B[0].length, C = zeros(n, m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { let s = 0;
      for (let p = 0; p < k; p++) s = f32(s + f32(f32(A[i][p]) * f32(B[p][j]))); C[i][j] = s; }
    return C; }
  function mmFP16pair(A, B) {
    const Ah = map2(A, v => half(f32(v))), Al = map2(A, (v, i, j) => half(f32(f32(v) - Ah[i][j])));
    const Bh = map2(B, v => half(f32(v))), Bl = map2(B, (v, i, j) => half(f32(f32(v) - Bh[i][j])));
    const n = A.length, k = B.length, m = B[0].length, C = zeros(n, m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { let s = 0;
      for (let p = 0; p < k; p++) s = f32(s + f32(Ah[i][p] * Bh[p][j] + Ah[i][p] * Bl[p][j] + Al[i][p] * Bh[p][j])); C[i][j] = s; }
    return C;
  }
  function mmInt(bits) { const L = Math.pow(2, bits - 1) - 1;
    return (A, B) => { const sa = maxAbs(A) || 1, sb = maxAbs(B) || 1;
      const Aq = map2(A, v => Math.round(v / sa * L)), Bq = map2(B, v => Math.round(v / sb * L));
      return map2(matmul(Aq, Bq), v => f32(v * sa * sb / L / L)); }; }
  // Analog behavioral chain (as mc_budget.py): wBits weights with per-cell gain, dBits duty per polarity
  // (two signed passes), static column offsets per pass, read noise per read, adcBits ADC (fitted or worst-case range).
  function mmAnalog(o, N) {
    const st = rng(1000 + (o.seed | 0)), nz = rng(5000 + (o.seed | 0));
    const G = zeros(N, N).map(r => r.map(() => 1 + (o.gain || 0) * st.gauss()));
    const op = Array.from({ length: N }, () => (o.offset || 0) * st.gauss());
    const on = Array.from({ length: N }, () => (o.offset || 0) * st.gauss());
    const Lw = Math.pow(2, (o.wBits || 8) - 1) - 1, Ld = Math.pow(2, o.dBits || 8) - 1;
    return (X, Y) => {
      const n = X.length, m = Y[0].length;
      const sw = maxAbs(Y) || 1, sa = maxAbs(X) || 1;
      const W = map2(Y, (v, i, j) => Math.round(v / sw * Lw) / Lw * G[i][j]);
      const FSw = Array.from({ length: m }, (_, j) => W.reduce((s, r) => s + Math.abs(r[j]), 0) + 1e-30);
      const out = zeros(n, m);
      for (const [sg, off] of [[1, op], [-1, on]]) {
        const dty = map2(X, v => Math.round(Math.max(sg * v, 0) / sa * Ld) / Ld);
        const Z = matmul(dty, W);
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
          Z[i][j] += off[j] * FSw[j]; if (o.noise) Z[i][j] += o.noise * FSw[j] * nz.gauss(); }
        for (let j = 0; j < m; j++) {
          let FS = FSw[j];
          if (o.fitted) { FS = 1e-30; for (let i = 0; i < n; i++) FS = Math.max(FS, Math.abs(Z[i][j])); }
          const step = 2 * FS / Math.pow(2, o.adcBits || 12);
          for (let i = 0; i < n; i++) out[i][j] += sg * Math.max(-FS, Math.min(FS, Math.round(Z[i][j] / step) * step));
        }
      }
      return map2(out, v => f32(v * sa * sw));
    };
  }
  function frontEnd(kind, opts, N) {
    if (kind === "fp64") return { mm: matmul, store: v => v };
    if (kind === "fp32") return { mm: mmF32, store: f32 };
    if (kind === "fp16pair") return { mm: mmFP16pair, store: f32 };
    if (kind === "int8") return { mm: mmInt(8), store: f32 };
    if (kind === "int") return { mm: mmInt(opts.bits || 10), store: f32 };
    if (kind === "analog") return { mm: mmAnalog(opts, N), store: f32 };
    throw new Error("unknown front end " + kind);
  }

  // ---------- SP2 purification (trace-correcting, stall rule, divergence guard) ----------
  function gershgorin(H) { let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < H.length; i++) { let r = 0; for (let j = 0; j < H.length; j++) if (j !== i) r += Math.abs(H[i][j]);
      lo = Math.min(lo, H[i][i] - r); hi = Math.max(hi, H[i][i] + r); } return [lo, hi]; }
  function sp2(H, no, fe, maxIter) {
    const N = H.length, [a, b] = gershgorin(H);
    let X = map2(H, (v, i, j) => fe.store(((i === j ? b : 0) - v) / (b - a)));
    const hist = []; let it = 0, diverged = false;
    for (it = 1; it <= (maxIter || 200); it++) {
      const X2 = map2(fe.mm(X, X), fe.store), t1 = trace(X), t2 = trace(X2);
      X = Math.abs(t2 - no) < Math.abs(2 * t1 - t2 - no) ? X2 : map2(X, (v, i, j) => fe.store(2 * v - X2[i][j]));
      if (!finite(X) || maxAbs(X) > 1e3) { diverged = true; break; }
      let sq = 0; for (const r of X) for (const v of r) sq += v * v;
      hist.push(Math.abs(trace(X) - sq));
      if (it > 4 && hist[hist.length - 1] >= hist[hist.length - 3]) break;
    }
    return { X, iters: it, hist, diverged };
  }

  // ---------- observables and refinement ----------
  function sensitivity(Co, Cv, eo, ev, form) { // dI/d(lower alpha_k), first-order perturbation theory
    const N = Co.length, no = eo.length, nv = ev.length, [[a, b], [c, d]] = form, s = new Array(N).fill(0);
    const M = zeros(no, nv);
    for (let i = 0; i < no; i++) for (let j = 0; j < nv; j++) {
      const B = 0.5 * (Co[a][i] * Cv[b][j] + Co[b][i] * Cv[a][j]) - 0.5 * (Co[c][i] * Cv[d][j] + Co[d][i] * Cv[c][j]);
      M[i][j] = B / (eo[i] - ev[j]); }
    for (let k = 0; k < N; k++) { let t = 0;
      for (let i = 0; i < no; i++) for (let j = 0; j < nv; j++) t += Co[k][i] * M[i][j] * Cv[k][j];
      s[k] = -4 * t; }
    return s;
  }
  function exact(H, no, form) {
    const { e, V } = eigh(H), N = H.length;
    const Co = cols(V, [...Array(no).keys()]), Cv = cols(V, [...Array(N - no).keys()].map(i => i + no));
    return { e, Co, Cv, eo: e.slice(0, no), ev: e.slice(no), map: sensitivity(Co, Cv, e.slice(0, no), e.slice(no), form) };
  }
  function oaStep(X, H, delta1) {
    const N = H.length, R = map2(matmul(T(X), X), (v, i, j) => (i === j ? 1 : 0) - v), S = matmul(T(X), matmul(H, X));
    const d = S.map((r, i) => r[i] / (1 - R[i][i]));
    const E = map2(S, (v, i, j) => i === j ? R[i][i] / 2 : (Math.abs(d[j] - d[i]) >= delta1 ? (v + d[j] * R[i][j]) / (d[j] - d[i]) : R[i][j] / 2));
    const XE = matmul(X, E); return { X: map2(X, (v, i, j) => v + XE[i][j]), d };
  }
  function relErr(s, ref) { let m = 0, r = 0;
    for (let k = 0; k < s.length; k++) { if (!Number.isFinite(s[k])) return Infinity; m = Math.max(m, Math.abs(s[k] - ref[k])); r = Math.max(r, Math.abs(ref[k])); }
    return r > 0 ? m / r : m; }
  // Range finder -> one Rayleigh-Ritz per block -> Ogita-Aishima steps (float64). Returns per-step errors.
  function refine(P, H, no, form, ref, o) {
    o = o || {}; const N = H.length, g = rng(o.seed ?? 7), kmax = o.kmax ?? 10, target = o.target ?? 1e-5;
    const Ro = zeros(N, no).map(r => r.map(() => g.gauss())), Rv = zeros(N, N - no).map(r => r.map(() => g.gauss()));
    let Co = qr(matmul(P, Ro));
    const Q = qr(matmul(map2(P, (v, i, j) => (i === j ? 1 : 0) - v), Rv));
    const CoTQ = matmul(T(Co), Q), CoCoTQ = matmul(Co, CoTQ);
    let Cv = qr(map2(Q, (v, i, j) => v - CoCoTQ[i][j]));
    const rr = C => { const { e, V } = eigh(matmul(T(C), matmul(H, C))); return { C: matmul(C, V), e }; };
    const o1 = rr(Co), v1 = rr(Cv);
    let X = hcat(o1.C, v1.C), d = o1.e.concat(v1.e);
    const steps = []; let reachedAt = null, last = null;
    for (let k = 0; k <= kmax; k++) {
      if (!finite(X) || maxAbs(X) > 1e3) { steps.push({ k, err: Infinity }); break; }
      const order = [...Array(N).keys()].sort((a, b) => d[a] - d[b]);
      const oi = order.slice(0, no), vi = order.slice(no);
      const cur = { Co: cols(X, oi), Cv: cols(X, vi), eo: oi.map(i => d[i]), ev: vi.map(i => d[i]) };
      const map = sensitivity(cur.Co, cur.Cv, cur.eo, cur.ev, form), err = relErr(map, ref);
      steps.push({ k, err }); last = Object.assign(cur, { map });
      if (err <= target && reachedAt === null) reachedAt = k;
      if (reachedAt !== null && k >= reachedAt + 1) break;
      const nx = oaStep(X, H, o.delta1 ?? 1e-10); X = nx.X; d = nx.d;
    }
    return { steps, reachedAt, last };
  }

  // ---------- rule table ----------
  function variants(spec) {
    const hasO = spec.atoms.some(a => a.t === "O="), hasOR = spec.atoms.some(a => a.t === "OR");
    const out = [];
    for (const hO of hasO ? [0.5, 1, 2] : [1]) for (const hOR of hasOR ? [1.5, 2, 2.5] : [2])
      for (const kOR of hasOR ? [0.6, 0.8, 1.0] : [0.8]) for (const kset of ["std", "2312"]) for (const g of [0.1, 0.3, 0.5])
        out.push({ p: { hO, hOR, kset, kOverride: { "C-OR": kOR } }, gamma: g });
    return out;
  }
  function leversExact(spec, gamma, p) {
    const { H, no } = buildH(spec, gamma, p), H0 = buildH(spec, 0, p).H;
    const x = exact(H, no, spec.form), x0 = exact(H0, no, spec.form);
    const q = rowsSq(x.Co), q0 = rowsSq(x0.Co);
    return { LA: x.map, LE: q.map((v, k) => v - q0[k]) };
  }
  function rowsSq(Co) { return Co.map(r => 2 * r.reduce((s, v) => s + v * v, 0)); }
  function ruleTable(spec, gamma, p, refined) {
    // refined: { Co, Cv, eo, ev, map } from the pipeline (used for the levers shown); robustness from exact variants.
    const N = spec.atoms.length, tol = 1e-9;
    const q = rowsSq(refined.Co), x0 = exact(buildH(spec, 0, p).H, refined.eo.length, spec.form), q0 = rowsSq(x0.Co);
    const LE = q.map((v, k) => v - q0[k]), LA = refined.map;
    const vs = variants(spec).map(v => leversExact(spec, v.gamma, v.p));
    const sgn = v => Math.abs(v) < tol ? 0 : Math.sign(v);
    const rows = spec.atoms.map((a, k) => {
      const s = sgn(LA[k]); const reach = a.t === "C" ? "through-space only" : "directed binding";
      let env = "no lever";
      if (s !== 0) env = a.t === "C" ? (s > 0 ? "withdrawing (positive charge nearby)" : "donating (negative charge nearby)")
        : (s > 0 ? "withdrawing (H-bond donor or Lewis acid on the lone pair)" : "donating (negative charge nearby; no H-bond donor on this lone pair)");
      const robust = vs.filter(v => sgn(v.LA[k]) === s).length / vs.length;
      return { k, name: a.name, frag: a.frag, type: a.t, LA: LA[k], LE: LE[k], sign: s, env, reach, robust,
               alsoBind: s !== 0 && sgn(LE[k]) === s };
    });
    for (const frag of ["diene", "dienophile"]) rows.filter(r => r.frag === frag)
      .sort((x, y) => Math.abs(y.LA) - Math.abs(x.LA)).forEach((r, i) => { r.rank = i + 1; });
    return rows;
  }
  function summary(rows, top) {
    const out = [];
    for (const frag of ["diene", "dienophile"]) {
      const fr = rows.filter(r => r.frag === frag && r.sign !== 0);
      const directed = fr.filter(r => r.reach === "directed binding");
      const space = fr.filter(r => r.reach !== "directed binding").sort((a, b) => a.rank - b.rank).slice(0, top || 2);
      for (const r of directed.concat(space)) out.push(r);
    }
    return out;
  }

  root.Engine = { presets, chain, buildH, eigh, exact, sp2, frontEnd, refine, ruleTable, summary, gershgorin,
                  KSETS, relErr, matmul, rng, half };
})(typeof window !== "undefined" ? window : globalThis);
