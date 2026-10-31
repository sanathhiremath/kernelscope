# KernelScope profiling

`profiling.py` provides a bounded `profile_model(model, inputs, output_dir)` callable. It records PyTorch CPU operators and, when available, CUDA device activity. It exports a Chrome/Perfetto-compatible `trace.json` and a machine-readable `operators.json` containing the top 25 operator groups, timings in microseconds, call counts, and memory allocation deltas.

```python
from profiling import profile_model

# Caller prepares the trained model and representative resident GPU inputs.
report = profile_model(model, images, 'artifacts/profiles/fp32')

# Match the benchmark's precision and memory layout explicitly.
report = profile_model(
    model, images, 'artifacts/profiles/fp16',
    context_factory=lambda: torch.autocast('cuda', dtype=torch.float16),
)
```

Call from the KernelScope directory or add it to Python's import path. A tensor or list is passed as a single argument, supporting detection models that accept `list[Tensor]`. Positional tuples and keyword dictionaries are supported. Default execution is two unrecorded warm-up calls and three profiled calls. Bounds limit each phase to 20 calls, and the model's original per-module training flags are restored afterward. Inputs and model placement are not modified by this helper. The caller must ensure concurrent GPU work is stopped before collecting comparable traces.

CPU-only runs are supported for validation. For CUDA inputs the report explicitly records whether CUDA device events were observed. Windows driver, CUPTI, PyTorch build, and profiler permissions can limit GPU tracing even when ordinary CUDA inference works. An exported trace alone does not establish successful GPU profiling: inspect `device_events_observed` and `warnings`. Missing CUDA events must be reported as a limitation, not replaced with claimed kernel measurements.

Open `trace.json` in a local trace viewer such as Perfetto to inspect the timeline. Trace files can reveal operator names, shapes, and environment details; review before publishing. No tensor values or Python stacks are intentionally recorded.

These traces locate operator work; they do not provide Nsight hardware counters, kernel occupancy, achieved memory bandwidth, or proof that a custom kernel is optimal. Profiled durations include instrumentation overhead and must not be compared directly with the uninstrumented CUDA-event benchmark. Inclusive operator times overlap; allocation deltas are not peak VRAM. Loading, preprocessing, transfers, and model quality require separate measurements.

Reference: [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html).
