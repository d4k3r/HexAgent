# Self-play throughput qualification V1

This is a resource-scheduling qualification for frozen Champion-2/Search-V2
NORMAL self-play, not a strength experiment. The workload fixes model identity,
game IDs/seeds, 128 visits, `c_puct=2.5`, parent-value-reduced FPU 0.25,
ACTIVE bridge control and LiteralWinner terminal authority.

Each topology partitions the same ID range into disjoint worker roots. Every
worker has its own lifecycle/lock/atomic commits and must pass the native-v2
postrun audit. The ONNX model and executable are read-only shared inputs.
Semantic fingerprints are compared explicitly: a changed CUDA batch shape may
change a neural near-tie, but it cannot bypass the full corpus audit.

The final 512-game confirmation compared the previous `1 x C64 / B96 / 200us`
baseline with `1 x C128 / B96 / 200us` on the tested Ryzen 7 9800X3D + RTX
5080 host.

| setting | games/s | Phase-A rows/s | simulations/s | mean batch |
|---|---:|---:|---:|---:|
| C64 / B96 / 200us | 1.1759 | 52.33 | 6698.7 | 36.98 |
| C128 / B96 / 200us | 1.7217 | 76.63 | 9808.2 | 63.54 |

The measured wall-clock improvement is 46.41%. Both runs had zero quarantine,
full certificate coverage and LiteralWinner agreement. C128 had the same
winner distribution, while 131/512 complete fingerprints differed (including
10 move sequences) through permitted CUDA near-tie effects; those differences
are reported, not treated as byte-identical equivalence.

The approved mass-self-play profile is one process, concurrency 128, max batch
96 and wait 200us. It is machine-specific qualified execution geometry. It
does not change Search-V2 semantics and does not automatically apply to paired
gameplay evaluation.
