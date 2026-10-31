"""Run KernelScope from a checkout on Windows or Ubuntu."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from kernel_scope.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
