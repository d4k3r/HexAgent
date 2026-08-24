# CONTROL teacher recipe

The final teacher-dose decision is **CONTROL**. This is a bounded causal
decision: validation diagnostics do not override paired gameplay.

Each candidate trains for four epochs of 400,000 base rows (batch 64; 6,250
optimizer steps per epoch) from the same parent tensor state. The selected
per-epoch source plan is:

| source | base rows | share |
|---|---:|---:|
| Teacher100 | 80,000 | 20% |
| Deep1600 | 0 | 0% |
| Historical salvage | 80,000 | 20% |
| fresh NORMAL | 180,000 | 45% |
| fresh FORCED | 60,000 | 15% |

Thus NORMAL/FORCED are 75%/25% inside the fresh component. Teacher100 and
historical salvage are static anchors; fresh sources are generated on-policy by
the current incumbent each generation.

DEEP5 replaced 4,096 Teacher100 rows with the fixed Deep1600 position set and
scored 0.5225 against CONTROL over 400 games, but its one-sided 95% bootstrap
LCB was 0.49. DEEP10 reused the same positions twice per epoch (8,192 rows),
scored 0.505, and had LCB 0.4775. Neither exceeded the precommitted strict
promotion threshold, so CONTROL is selected. This does not establish that
Deep1600 has no value in other data or dose regimes.

Teacher and Deep sources use normalized physical root visits for the 121-action
soft policy target. Student value supervision remains the verified source-game
side-relative outcome `z`; teacher utility fields are retained only as
provenance. Colour-transpose augmentation is trainer-side.
