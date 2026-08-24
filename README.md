# HexAgent - reconstructed Hex policy/value agent

This is an independent post-coursework reconstruction and research portfolio.
The University of Manchester COMP34111 Group 49 work is historical context
only: this repository is neither a fork nor a wholesale republication of that
coursework, KataHex, Benzene, or Neurobenzene.

## System

The modern reconstruction uses deterministic Python and C++ PUCT around a
six-plane Student policy/value boundary. The C++ layer has optional ONNX
Runtime CPU/CUDA integration and shared batching, but this repository bundles
no vendor runtime, model, checkpoint, or production binary. CPU-only tests and
the no-runtime CMake core are the supported public reproduction path.

Literal board connectivity is always terminal authority. The elementary
connection certificate and physical realizer are restricted audited mechanisms,
not general H-search.

## Qualified milestones

- Stage-7 certificate-aware salvage retains no policy rows after the first
  valid restricted certificate.
- Champion-1 is a historical matched-evaluation lineage, not the current
  champion.
- Native self-play-v2 has explicit new, resume and completed-run lifecycle
  semantics.
- Search-V2 is 128 visits, `c_puct=2.5`, parent-value-reduced FPU, reduction
  0.25. It is a qualified operating point, not a strongest-final-play claim.
- Frozen forced-prefix diversity improved gameplay in the matched BASE versus
  DIVERSE experiment: 220-180, score 0.55, one-sided bootstrap 95% LCB 0.52.
  DIVERSE then beat Champion-1 261-139, score 0.6525, LCB 0.62, supporting
  Champion-2 promotion.
- Deep-Teacher-1600 completed generation and frozen-subset convergence. It is
  stability evidence for a causal teacher-dose experiment, not Student-strength
  evidence. Neither DEEP5 nor DEEP10 passed the frozen paired-game promotion
  gate against CONTROL, so CONTROL is the selected production teacher recipe.
- A controlled 512-game resource qualification selected one-process
  C128/B96/wait-200us for mass self-play on the tested Ryzen 7 9800X3D + RTX
  5080 machine: 1.7217 games/s versus 1.1759 for C64/B96/wait-200us, a 46.41%
  improvement. This changes execution geometry only, not Search-V2 or gameplay
  evaluation settings.
- The fail-closed autonomous production coordinator completed one real
  generation end-to-end. Its S4903 challenger beat Champion-2 249-151,
  score 0.6225, one-sided bootstrap 95% LCB 0.595, and was manually ratified
  as Champion-3. The prepared next three-generation run has not been launched
  in this public snapshot.

This work does not claim that Champion-3 beats KataHex or MoHex, that 11x11
Hex is solved, that 1600 is optimal, that CONTROL is globally optimal, that
C128 is optimal on every machine, or that autonomous improvement is guaranteed.

## Reproduce bounded checks

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
    cmake -S cpp -B build/cpp
    cmake --build build/cpp
    ctest --test-dir build/cpp --output-on-failure

See `docs/` for scope, reproduction and milestone detail. `results/` contains
only compact aggregate evidence; raw games, teacher records, weights and local
runtime bundles are deliberately excluded. The production coordinator is
published for inspection and CPU/fake lifecycle testing; its real adapters
expect locally supplied, separately qualified model/runtime artifacts.

## Attribution and boundary

KataHex was an external teacher/reference system. `board.py` includes a
behavior-preserving virtual-detector port; its scope and preserved upstream
license text are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
No new owned-code licence is asserted by this migration.
