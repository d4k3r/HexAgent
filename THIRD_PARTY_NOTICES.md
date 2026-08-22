# Third-party notices

## KataHex

KataHex is a third-party project by David J Wu (lightvector) and other
authors. Its applicable upstream license text is preserved verbatim in
[`licenses/KataHex-LICENSE.txt`](licenses/KataHex-LICENSE.txt).

The virtual-terminal detector in
`src/hex_reconstruction/board.py` is a behavior-preserving port of the
connection-checking logic audited from KataHex revision
`41a65784fac932046eb3350662d4f1eca1b810b3`, specifically
`cpp/game/gamelogic.cpp`. It remains deliberately separate from this
repository’s literal connectivity/DSU terminal authority.

`cpp/src/connection_certificate.cpp` uses elementary bridge and edge-bridge
patterns informed by that audited KataHex behavior, expressed as a separately
implemented restricted certificate/realizer design. It is not represented as a
copy of the KataHex source tree or as a general H-search implementation.

The remaining independent reconstruction is not thereby labelled KataHex
code. No KataHex source tree, binary, model, or other bundled dependency is
included here.

## Other references

Benzene and Neurobenzene were inspected as references only. Their source and
patch trees are not included in this repository.
