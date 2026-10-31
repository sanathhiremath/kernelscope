import argparse
import json
from pathlib import Path
from .benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description='KernelScope pretrained detector GPU benchmark')
    parser.add_argument('command', choices=['demo'], nargs='?', default='demo')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'artifacts/demo.json')
    args = parser.parse_args()
    report = benchmark(args.iterations, args.repeats, args.warmup, tuple(args.batch_sizes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(args.output)
    return 0
