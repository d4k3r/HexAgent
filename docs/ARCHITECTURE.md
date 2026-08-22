# Qualified reconstruction core

This repository contains an independent reconstruction of a compact 11x11
Hex policy/value system. The public core has deterministic Python and C++ PUCT
implementations, a six-plane Student boundary, symmetry and swap controls, and
optional ONNX Runtime integration. Vendor runtimes, models, checkpoints and
generated runs are intentionally not included.

The C++ project configures a CPU-only core by default. Optional ONNX Runtime
CPU/CUDA targets require an externally supplied runtime path; they are
qualification interfaces, not a bundled deployment.

Terminal win authority remains the board physical connectivity predicate.
Later certificate and realizer components are deliberately restricted aids,
not a general H-search implementation.
