"""Bounded PyTorch operator profiling, separate from benchmark timing.

The caller owns device placement and input preprocessing. This module never
downloads weights or moves a model to another device.
"""

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Callable, Mapping

import torch


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


def _invoke(model, inputs):
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def profile_model(
    model,
    inputs,
    output_dir,
    *,
    iterations: int = 3,
    warmup: int = 2,
    top_n: int = 25,
    context_factory: Callable = nullcontext,
):
    """Export ``trace.json`` and ``operators.json`` and return the report.

    Inputs may be one tensor/list, a positional argument tuple, or keyword dict.
    Pass a fresh-context factory to profile AMP, e.g.
    ``lambda: torch.autocast('cuda', dtype=torch.float16)``. Profiling defaults to
    the caller's dtype and layout. The model temporarily enters evaluation mode;
    each module's original training flag is restored even if inference fails.

    CUDA activities are requested only for CUDA inputs and when the installed
    profiler advertises support. Absence of observed CUDA events is explicit in
    the report; a CPU-only trace must not be presented as GPU kernel profiling.
    """
    if not isinstance(iterations, int) or not 1 <= iterations <= 20:
        raise ValueError('iterations must be an integer from 1 to 20')
    if not isinstance(warmup, int) or not 0 <= warmup <= 20:
        raise ValueError('warmup must be an integer from 0 to 20')
    if not isinstance(top_n, int) or not 1 <= top_n <= 100:
        raise ValueError('top_n must be an integer from 1 to 100')

    input_tensors = list(_tensors(inputs))
    cuda_devices = {tensor.device for tensor in input_tensors if tensor.is_cuda}
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_requested = bool(cuda_devices)
    cuda_supported = torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
    if cuda_requested and cuda_supported:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def synchronize():
        for device in cuda_devices:
            torch.cuda.synchronize(device)

    training_flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode(), context_factory():
            for _ in range(warmup):
                _invoke(model, inputs)
            synchronize()
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                for _ in range(iterations):
                    with torch.profiler.record_function('KernelScope inference'):
                        _invoke(model, inputs)
                synchronize()
    finally:
        for module, training in training_flags:
            module.training = training

    events = profiler.events()
    cuda_event_count = sum('CUDA' in str(event.device_type).upper() for event in events)
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append({
            'operator': event.key,
            'input_shapes': event.input_shapes,
            'calls': event.count,
            'self_cpu_us': event.self_cpu_time_total,
            'total_cpu_us': event.cpu_time_total,
            'self_device_us': event.self_device_time_total,
            'total_device_us': event.device_time_total,
            'self_cpu_memory_bytes': event.self_cpu_memory_usage,
            'self_device_memory_bytes': event.self_device_memory_usage,
        })
    sort_key = 'self_device_us' if cuda_event_count else 'self_cpu_us'
    rows.sort(key=lambda row: row[sort_key], reverse=True)
    warnings = []
    if cuda_requested and not cuda_supported:
        warnings.append('This PyTorch profiler build does not advertise CUDA activity support.')
    if cuda_requested and not cuda_event_count:
        warnings.append('No CUDA device events were observed; do not infer GPU kernel bottlenecks from this trace.')

    report = {
        'version': '0.1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'torch_version': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'platform': platform.platform(),
        'iterations': iterations,
        'warmup': warmup,
        'requested_activities': [str(activity).split('.')[-1] for activity in activities],
        'cuda_inputs': cuda_requested,
        'cuda_event_count': cuda_event_count,
        'device_events_observed': bool(cuda_event_count),
        'sort_metric': sort_key,
        'operator_groups_total': len(rows),
        'top_operators': rows[:top_n],
        'trace_file': 'trace.json',
        'summary_file': 'operators.json',
        'warnings': warnings,
        'scope': 'Instrumented inference with resident inputs. Profiling overhead invalidates comparisons with uninstrumented benchmark latency. Operator totals may include nested work and are not additive. Memory columns are allocation deltas, not process VRAM or peak memory. No preprocessing, transfers, hardware counters, occupancy, or task-accuracy measurements.',
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(destination / 'trace.json'))
    (destination / 'operators.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
