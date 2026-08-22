# Public export manifest

This repository is a deliberately small public export of independent post-coursework reconstruction work. It is not a fork or mirror of the University of Manchester coursework repository, KataHex, or any other historical repository.

## Included

| Category | Public status | Rationale |
| --- | --- | --- |
| `src/hex_reconstruction/board.py` | Safe to publish | Independently reconstructed 11x11 board, literal terminal check, six-plane feature encoding, and coordinate helpers. |
| `schema.py`, `validation.py`, `manifest.py`, `soft_policy.py` | Safe to publish | New data-contract, validation, resumability, and soft-policy-loss work. |
| `student_training.py` | Safe to publish | Reconstructed PyTorch policy/value Student architecture and training primitives. |
| `gtp.py` | Safe to publish | New framed GTP client/parser; it communicates with an external engine but includes no KataHex source. |
| Focused unit tests and fake GTP engine | Safe to publish | New test infrastructure for the reconstructed components. |
| `results/` | Generated result safe to publish | Compact, derived JSON/CSV/SVG summaries only; no raw corpus, checkpoints, model weights, or engine logs. |

## Excluded

| Category | Status | Reason |
| --- | --- | --- |
| `HexAgent/` | Historical/university reference only | Contains university framework and multiple student groups; it is not appropriate to republish wholesale. |
| `katahex/` | Third-party reference only | External engine source with separate provenance/licensing. |
| KataHex b28 model, Student checkpoints, ONNX files | Excluded | Model redistribution and checkpoint provenance are not established for this portfolio snapshot. |
| Teacher corpora, raw GTP logs, WSL toolchains/builds | Excluded | Large, environment-specific, and/or include external execution artifacts. |
| Historical MoHex/Neurobenzene work | Reference only | External/historical dependency work; out of scope for this focused portfolio export. |
| University `Game.py`, `AgentBase`, C++ PUCT path | Excluded | University-framework integration is not republished here. |

No license is included in this initial snapshot because the public subset is intentionally conservative while ownership and redistribution scope are being reviewed.

## Migration note (2026-08-22)

The original manifest above is retained as historical evidence of the
13-August curation boundary. The current curated migration expands the public
independent reconstruction to qualified C++/Python PUCT, restricted
certificate/realizer, self-play-v2, Search-V2, forced-prefix diversity,
Champion-2 and Deep-Teacher convergence infrastructure. Version-2 structured
provenance is `EXPORT_SOURCES_V2.json`; it deliberately excludes raw corpora,
teacher banks, models, build/runtime trees and local absolute-path registries.
Third-party scope and the preserved KataHex upstream license text are recorded
in `THIRD_PARTY_NOTICES.md` and `licenses/KataHex-LICENSE.txt`.
