# Catalyst Precision (MolMAC branch `cataline`)

**How much numerical precision does a cheap or analog chip need to help design catalysts?**

This project takes a simple quantum-chemistry model of a Diels-Alder reaction (a diene meeting a
dienophile), solves it first on a deliberately imprecise "front end", finishes the job with an exact
digital step, and measures how rough the front end is allowed to be. From the result it produces a short
table of what an ideal catalyst would need to do at each atom.

It lives in the `catalyst/` folder and extends MolMAC, an analog-array approach to molecular energy
calculations. Nothing outside this folder is changed.

## What we found

- The imprecise front end is **necessary**: starting the exact step from scratch fails in nearly every run.
- The precision it needs **grows with molecule size**. Over 8 to 64 π-electron centres, a digital integer
  front end needs 8 to 12 bits, and a modelled analog chip needs 10 to 14 bits.
- **Read noise** is the error source that limits the design most.
- **No energy saving.** The exact finishing step costs about as much as solving the problem directly,
  so this design is a precision study, not an efficiency claim.
- We do not claim that analog hardware is required for this problem.

## Try it

Open `catalyst/frontend/catalyst_pipeline.html` in a browser. It walks through molecule, processing, ideal-catalyst
characteristics, and a readable output. No install needed.

To run the Python code:

```
pip install numpy scipy
cd catalyst
python h0_exact.py        # exact checks, under a second
python rules.py           # catalyst requirement table
python scale.py           # precision vs molecule size
```

Scripts for the enzyme-screening check (`align_ident.py`, `background.py`, `positive_control.py`,
`matched.py`) also need `pandas`, `scikit-learn` and `biopython`, plus your own dataset CSV passed as the
first argument.

## What is in the repo

| Folder | Contents |
| --- | --- |
| `frontend/` | The interactive page and its source |
| `*.py` (this folder) | All Python scripts; each states its predictions in its header |
| `docs/` | Working notes (`NOTES.md`). A rigorous technical write-up will go here |

## Status and limits

Research prototype. The molecular model is a simplified Hückel model, the analog chip is a simulation,
and the catalyst is represented by a single on-site energy shift. Results are for that model only.
Sign conventions, error bars and caveats are in `docs/NOTES.md` for now.

## Author

Anirudh Dutta
