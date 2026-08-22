# Native self-play v2

The native runner distinguishes a new run, a resumable partial run, and a
completed run. Its public scripts validate lifecycle metadata and deterministic
game identity before treating output as complete. Production corpora and run
directories are intentionally excluded.
