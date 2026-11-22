require("./engine.js"); const E = globalThis.Engine, X = require("./expected.json");
const mx = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const P = E.presets();
// 1) dE_int butadiene + ethylene at gamma = 1
let s = P["butadiene + ethylene"], a = E.buildH(s, 1), b = E.buildH(s, 0);
const ea = E.eigh(a.H).e, eb = E.eigh(b.H).e, sum = (e, n) => e.slice(0, n).reduce((x, y) => x + y, 0);
console.log("dE_int:", (-2 * sum(ea, a.no) + 2 * sum(eb, b.no)).toFixed(6), "expected", X.dEint.toFixed(6));
// 2) acrolein sensitivity map, exact path
s = P["butadiene + acrolein"]; a = E.buildH(s, 0.3);
console.log("acrolein map max diff:", mx(E.exact(a.H, a.no, s.form).map, X.map_acrolein).toExponential(2));
// 3) chain N = 24 vs scale.py
s = E.chain(12, 12, true); a = E.buildH(s, 0.3);
console.log("chain N=24 map max diff:", mx(E.exact(a.H, a.no, s.form).map, X.map_chain24).toExponential(2));
// 4) rule-table levers and robustness vs rules.py (refined = exact here)
for (const name of Object.keys(P)) { s = P[name]; a = E.buildH(s, 0.3); const x = E.exact(a.H, a.no, s.form);
  const rows = E.ruleTable(s, 0.3, {}, x);
  console.log(name, "LA", mx(rows.map(r => r.LA), X[name].LA).toExponential(1), "LE", mx(rows.map(r => r.LE), X[name].LE).toExponential(1),
              "robust", mx(rows.map(r => r.robust), X[name].robust).toFixed(3)); }
// 5) full pipeline: each front end on acrolein and on N = 24
for (const [lab, spec] of [["acrolein", P["butadiene + acrolein"]], ["chain24", E.chain(12, 12, true)]]) {
  a = E.buildH(spec, 0.3); const ref = E.exact(a.H, a.no, spec.form).map;
  for (const [k, o] of [["fp64", {}], ["fp32", {}], ["fp16pair", {}], ["int8", {}], ["analog", { adcBits: 12, fitted: true, seed: 3 }],
                        ["analog", { adcBits: 8, fitted: false, seed: 3 }], ["analog", { adcBits: 12, fitted: true, noise: 1e-3, gain: 0.017, seed: 3 }]]) {
    const t0 = Date.now(), fe = E.frontEnd(k, o, a.H.length), sp = E.sp2(a.H, a.no, fe);
    const r = E.refine(sp.X, a.H, a.no, spec.form, ref, { seed: 7 });
    console.log(`${lab} ${k} ${JSON.stringify(o)}: SP2 ${sp.iters} it, reached at k=${r.reachedAt}, errs ${r.steps.map(x => x.err.toExponential(0)).join(" ")} (${Date.now() - t0} ms)`);
  } }
