# Bounded reproduction

Install the Python dependencies in an isolated environment, then run the
public CPU unit tests:

    PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
    cmake -S cpp -B build/cpp
    cmake --build build/cpp
    ctest --test-dir build/cpp --output-on-failure

The optional ONNX CPU/CUDA paths need an externally installed compatible ONNX
Runtime and an intentionally supplied model. No model, runtime bundle, raw
self-play corpus, teacher bank, or production workload is required by the
bounded public checks.
