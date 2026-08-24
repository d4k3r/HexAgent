# Controlled reconstruction results

The authoritative machine-readable summaries are in `results/`.

## Teacher-data scaling

Five paired seeds used the same 384-game / 20,082-position game-level validation split. Canonical training used 1,536 games / 81,809 Phase-A rows; expanded training added 2,200 qualified teacher games for 3,736 games / 198,587 rows.

Best validation policy CE within runs capped at 24 epochs was 1.7308 for canonical training and 1.5909 for expanded training. KL was 0.5348 and 0.3950 respectively. Expanded CE and KL improved in all five paired seeds. An approximate optimizer-step comparison (canonical epoch 24, 7,680 steps; expanded epoch 10, 7,760 steps) also favoured expanded training.

These are policy-distillation results, not a claim of gameplay strength or statistical significance.

## On-policy teacher diagnostic

A deterministic sample of 184 raw-policy disagreement positions was re-queried with the same qualified KataHex b28 configuration. Nineteen positions were already teacher virtual/proof terminal, leaving 165 searchable positions. On those positions, best-validation-policy expanded Students had lower teacher CE (3.0269 vs 3.2342), lower KL (1.5228 vs 1.7301), and higher teacher top-1 agreement (36.36% vs 25.45%) than canonical Students.

Raw deterministic argmax games changed ranking under checkpoint-selection rules, so they are treated as diagnostics of greedy-policy behaviour rather than a full-agent benchmark.

## Teacher-dose decision

CONTROL is the selected production recipe. It holds the teacher contribution at
80,000 base rows per epoch and uses 80,000 Teacher100, 80,000 historical
salvage, 180,000 fresh NORMAL and 60,000 fresh FORCED rows. DEEP5 replaced
4,096 Teacher100 rows with the frozen Deep1600 positions; DEEP10 replaced
8,192 rows, reusing every one of the 4,096 positions twice per epoch.

Neither candidate passed the precommitted paired-game promotion gate against
CONTROL: DEEP5 scored 209-191 (0.5225; one-sided 95% LCB 0.49), while DEEP10
scored 202-198 (0.505; LCB 0.4775). These results select CONTROL for production
without claiming that deep teacher targets are generally unhelpful.

## Throughput and autonomous generation

On the explicitly tested Ryzen 7 9800X3D + RTX 5080 configuration, a 512-game
mass-self-play confirmation measured 1.7217 games/s for one process with
concurrency 128, max batch 96 and wait 200us, compared with 1.1759 games/s for
the C64/B96/200us baseline: +46.41%. Both configurations completed semantic
audits; C128 is a hardware-qualified execution profile, not a search-strength
claim or a gameplay-evaluation default.

The real autonomous Generation-1 CONTROL run completed from Champion-2. Its
S4903 challenger won 249-151 over 200 paired openings (score 0.6225, LCB
0.595), passed the promotion gate, and was manually ratified as Champion-3.
The next three-generation CONTROL run is prepared but intentionally not
represented here as started or completed.
