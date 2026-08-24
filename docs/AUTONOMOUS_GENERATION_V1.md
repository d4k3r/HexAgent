# Autonomous generation V1

The production coordinator is a thin, fail-closed layer over the qualified
native self-play, corpus audit, preparation, Stage8B training, ONNX parity and
paired-gameplay executors. It publishes code and configuration, but not the
private models, corpora, runtime bundles or production outputs required to run
the real adapters.

The durable stages are `SELFPLAY_NORMAL`, `SELFPLAY_FORCED`,
`SELFPLAY_AUDIT`, `DATA_PREP`, `TRAIN_CANDIDATES`, `EXPORT_CANDIDATES`,
`CANDIDATE_SCREEN`, `PROMOTION_MATCH` and `PROMOTION_DECISION`. A run records
immutable run and generation manifests, atomic mutable state, append-only
attempt history, stage evidence and summaries. Resume revalidates identity and
evidence and never guesses completion from directory existence.

An approved recipe binds the source mixture, Search-V2, forced-prefix bank,
resource profile, parent identity, candidate seeds, pair counts and promotion
gate. Heavy stages take a nonblocking local machine lease; the coordinator
never kills another owner. Corruption, source/model/search mismatch, failed
parity, invalid paired evidence or executor failure stops the run and preserves
evidence. A normal challenger loss retains the incumbent and continues.

Promotion advances a hash-bound **run-local** incumbent for the next
generation. The global Champion registry remains read-only during an unattended
run; later official ratification is manual. The human-readable monitor only
reads completed/status files and formats durations as `7m 12s` or `1h 08m`.

The selected production recipe is CONTROL with Search-V2 and the separate
mass-self-play C128/B96/200us resource profile. It has three seed-only
candidates per generation. The first real generation completed end-to-end;
its successful run-local challenger was later manually ratified as Champion-3.
The prepared next three-generation run is not included as an execution claim in
this snapshot.
