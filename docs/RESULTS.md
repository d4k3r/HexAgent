# Controlled reconstruction results

The authoritative machine-readable summaries are in `results/`.

## Teacher-data scaling

Five paired seeds used the same 384-game / 20,082-position game-level validation split. Canonical training used 1,536 games / 81,809 Phase-A rows; expanded training added 2,200 qualified teacher games for 3,736 games / 198,587 rows.

Best validation policy CE within runs capped at 24 epochs was 1.7308 for canonical training and 1.5909 for expanded training. KL was 0.5348 and 0.3950 respectively. Expanded CE and KL improved in all five paired seeds. An approximate optimizer-step comparison (canonical epoch 24, 7,680 steps; expanded epoch 10, 7,760 steps) also favoured expanded training.

These are policy-distillation results, not a claim of gameplay strength or statistical significance.

## On-policy teacher diagnostic

A deterministic sample of 184 raw-policy disagreement positions was re-queried with the same qualified KataHex b28 configuration. Nineteen positions were already teacher virtual/proof terminal, leaving 165 searchable positions. On those positions, best-validation-policy expanded Students had lower teacher CE (3.0269 vs 3.2342), lower KL (1.5228 vs 1.7301), and higher teacher top-1 agreement (36.36% vs 25.45%) than canonical Students.

Raw deterministic argmax games changed ranking under checkpoint-selection rules, so they are treated as diagnostics of greedy-policy behaviour rather than a full-agent benchmark.
