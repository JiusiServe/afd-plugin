# Running NPU unit tests

NPU unit tests depend on vLLM and vLLM-Ascend. Run them in an Ascend development
environment that uses the versions pinned by this repository.

From the repository root, run:

```bash
python3 -m pytest -q tests/unit -m "not gpu and not vllm_runtime"
```

The suite covers the NPU worker and model-runner contracts, CAM connector
behavior, and module isolation. Tests that require unavailable NPU dependencies
are skipped in CPU-only CI, so a passing CPU run does not replace this check.

`tests/conftest.py` establishes the vLLM-Ascend import order required by the
test suite. No manual module preloading is needed.

When diagnosing a failure, run the failing pytest node by itself first. A test
that passes alone but fails in the full suite usually indicates leaked module
or monkeypatch state. Test fixtures and mocks must also follow the concrete
vLLM types and signatures used by the pinned runtime.
