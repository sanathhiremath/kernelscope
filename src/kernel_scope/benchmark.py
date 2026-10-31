"""Randomized repeated GPU-resident inference timing and output preservation."""
import copy
import random
import time
from statistics import median
import torch
from .quality import agreement
from .workload import load_workload, serialized_predictions


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered)-1)*quantile
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def benchmark(iterations=10, repeats=3, warmup=3, batch_sizes=(1, 4, 8)):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and NVIDIA GPU required; no CPU fallback results are fabricated.')
    if min(iterations, repeats, warmup) < 1 or not batch_sizes or min(batch_sizes) < 1:
        raise ValueError('Iterations, repeats, warmup and batch sizes must be positive.')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    cpu_model, cpu_inputs, provenance = load_workload(max(batch_sizes))
    inputs = [image.cuda() for image in cpu_inputs]
    references = {}
    model = copy.deepcopy(cpu_model).cuda()
    with torch.inference_mode():
        for batch_size in batch_sizes:
            references[batch_size] = serialized_predictions(model(inputs[:batch_size]))
    del model
    torch.cuda.empty_cache()
    configurations = [(name, mixed, layout, size) for size in batch_sizes for name, mixed, layout in
                      [('FP32', False, False), ('FP16', True, False), ('FP16 channels last', True, True)]]
    measurements = {(name, size): dict(samples_ms=[], trials=[], comparisons=[]) for name, _, _, size in configurations}
    randomizer = random.Random(42)
    orders = []
    with torch.inference_mode():
        for repeat in range(repeats):
            order = configurations.copy()
            randomizer.shuffle(order)
            orders.append([f'{name} batch {size}' for name, _, _, size in order])
            for name, mixed, layout, size in order:
                model = copy.deepcopy(cpu_model).cuda()
                if layout:
                    model.to(memory_format=torch.channels_last)
                    model.backbone.register_forward_pre_hook(
                        lambda module, args: (args[0].contiguous(memory_format=torch.channels_last),))
                batch = inputs[:size]
                with torch.autocast('cuda', dtype=torch.float16, enabled=mixed):
                    for _ in range(warmup):
                        outputs = model(batch)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    samples, wall_samples = [], []
                    for _ in range(iterations):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wall_start = time.perf_counter()
                        start.record()
                        outputs = model(batch)
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end))
                        wall_samples.append((time.perf_counter()-wall_start)*1000)
                peak = torch.cuda.max_memory_allocated()
                raw_finite = all(bool(torch.isfinite(output[field]).all().item())
                                 for output in outputs for field in ('boxes', 'scores'))
                predictions = serialized_predictions(outputs)
                check = agreement(references[size], predictions)
                check['raw_outputs_finite'] = raw_finite
                check['passed'] = check['passed'] and raw_finite
                record = measurements[name, size]
                record['samples_ms'].extend(samples)
                record['comparisons'].append(check)
                record['trials'].append(dict(repeat=repeat, median_batch_ms=median(samples),
                                             median_synchronized_wall_ms=median(wall_samples),
                                             peak_allocated_bytes=peak))
                print(f'{name}, batch {size}, trial {repeat+1}: {median(samples):.2f} ms; preservation {check["preservation_f1"]:.3f}', flush=True)
                del outputs, model
                torch.cuda.empty_cache()
    results = []
    for name, _, _, size in configurations:
        row = measurements[name, size]
        comparisons = row.pop('comparisons')
        row.update(name=f'{name} · batch {size}', variant=name, batch_size=size, median_batch_ms=median(row['samples_ms']),
                   p95_batch_ms=percentile(row['samples_ms'], .95),
                   images_per_second=1000*size/median(row['samples_ms']),
                   peak_allocated_bytes=max(trial['peak_allocated_bytes'] for trial in row['trials']),
                   numerical_gate_passed=all(check['passed'] for check in comparisons),
                   detection_agreement=min(check['preservation_f1'] for check in comparisons),
                   detection_agreement_details=comparisons)
        results.append(row)
    return dict(version='0.1', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                cuda=torch.version.cuda, iterations=iterations, repeats=repeats, warmup=warmup,
                seed=42, trial_orders=orders, results=results, provenance=provenance,
                comparison_definition='All-class detections score >=0.3; greedy same-class IoU >=0.9 matching against FP32 outputs of same batch. Gate requires finite outputs, nonempty reference and preservation F1 >=0.95 in every trial. This is output preservation, not annotated task accuracy.',
                scope=f'Frozen pretrained model, {len(inputs)} correlated frames from one licensed video. GPU-resident input tensors; detection transform and postprocessing included, image decode, host-to-device transfer and loading excluded. CUDA-event duration can include CPU launch gaps; synchronized wall medians also retained. Memory is total PyTorch peak allocated, not total GPU memory. Laptop power/thermal/background processes can affect results.')
