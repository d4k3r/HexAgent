# HexAgent — reconstructed Hex policy/value agent

This repository is my independent post-coursework reconstruction and experimental extension of work that began in a University of Manchester AI & Games group project. It is **not** a fork or wholesale republication of the original coursework repository. The original project included university-provided infrastructure and multiple student groups; this public repository contains only a conservative subset of my later reconstruction tooling and results.

## Architecture

The intended playing system is:

```text
state → Student policy P(s,a) + value V(s) → PUCT-guided MCTS → root visits → move
```

The currently qualified live gameplay diagnostic is deliberately narrower:

```text
state → Student → legal mask → argmax move
```

It has **no MCTS/PUCT**, so its outcomes are not full-agent strength results.

## KataHex teacher and data contract

The Student distils completed KataHex root-search **physical visit distributions**, not raw neural-network output or one-hot played moves. The policy head remains 121 physical board actions: pass, resign, and swap are not policy actions.

Teacher games have two explicit phases:

- **Phase A:** completed KataHex root visits supervise a soft policy target `π`; the root value is retained separately.
- **Phase B:** after KataHex virtual terminal but before the university-style literal board connection, pass-forbidden physical completion continues to literal termination. These rows have `π = null` and zero policy weight.

## Dataset and controlled scaling result

The qualified teacher corpus contains **4,120 complete games** and **218,669 Phase-A supervised positions**.

For a controlled scaling comparison, training used the same frozen validation split in every run:

| Split | Games | Phase-A rows |
| --- | ---: | ---: |
| Canonical training | 1,536 | 81,809 |
| Expanded training | 3,736 | 198,587 |
| Shared validation | 384 | 20,082 |

Across five paired seeds, best validation performance within runs capped at 24 epochs improved with the expanded teacher data:

| Metric | Canonical mean | Expanded mean |
| --- | ---: | ---: |
| Policy cross-entropy | 1.7308 | 1.5909 |
| Policy KL | 0.5348 | 0.3950 |

Expanded CE and KL improved in 5/5 paired seeds. The improvement also persisted in an approximately matched optimizer-step comparison. This is not a statistical-significance claim.

## Current finding

On 184 deterministic Student-generated disagreement states re-queried from KataHex, 165 were searchable and 19 were already teacher virtual/proof terminal. On searchable states, best-validation-policy expanded Students were more aligned with the teacher than canonical Students: CE 3.0269 vs 3.2342 and teacher top-1 agreement 36.36% vs 25.45%.

However, deterministic raw-policy gameplay ranking changed when selecting final rather than best-policy checkpoints. I therefore treat raw greedy games as a diagnostic of policy behaviour, not a stable full-agent benchmark. The next meaningful evaluation is Student policy/value integrated inside qualified PUCT/MCTS search.

## Validation notes

- Validation is held out by game/trajectory, not by independently sampled positions.
- Exact encoded Student-input overlap was measured at 4.01% for canonical train vs validation and 5.36% for expanded train vs the same validation set.
- Current value-head metrics are affected by a strong physical-colour / side-to-move shortcut.
- Full reconstructed Student+PUCT playing-strength evaluation remains future work.

## Repository map

- `src/hex_reconstruction/` — board/features, versioned example schema, soft-policy loss, Student model, GTP framing, and resumable manifests.
- `tests/` — focused offline tests, including fake-engine protocol tests.
- `results/` — compact paired-scaling metrics, overlap audit, and SVG plots.
- `docs/RESULTS.md` — concise result interpretation.
- `provenance/PUBLIC_MANIFEST.md` — publication scope and exclusions.

## Lightweight reproduction

The public subset has no teacher model, checkpoints, engine binary, university framework, or generated corpus. It can reproduce its focused unit tests after installing the declared Python dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Full teacher generation and scaled training require external KataHex assets and the frozen local dataset/checkpoint artifacts, intentionally excluded here.

## Attribution

KataHex is an external teacher/reference system. The University of Manchester framework and historical Group 49 coursework code remain separate historical references and are not included here. This repository does not imply authorship of KataHex, university infrastructure, or all historical Group 49 work.
