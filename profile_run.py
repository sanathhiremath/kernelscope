"""Profile the frozen real detector separately from timing trials."""
import sys
from pathlib import Path
import torch
from profiling import profile_model

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from kernel_scope.workload import load_workload

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this workload.')
    model, images, _ = load_workload(1)
    report = profile_model(model.cuda(), [image.cuda() for image in images],
                           Path(__file__).parent / 'artifacts/profile')
    print('CUDA events:', report['cuda_event_count'])
    print('Warnings:', report['warnings'])
