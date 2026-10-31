# KernelScope

GPU inference optimization on an NVIDIA RTX 3050 6 GB laptop. The modular benchmark runs a frozen pretrained SSDLite320 MobileNetV3 Large detector on attributed FrameTrace street-video images. It compares FP32, FP16 autocast, and FP16 with channels-last backbone layout.

## Run

For separate GitHub clones, keep FrameTrace and KernelScope as sibling directories:

```sh
git clone https://github.com/sanathhiremath/frametrace.git
git clone https://github.com/sanathhiremath/kernelscope.git
cd frametrace
python run.py prepare-real
python run.py demo --real
cd ../kernelscope
python run.py demo --iterations 10 --repeats 3 --warmup 5 --batch-sizes 1 4
python profile_run.py
```

Install each project's dependencies in an isolated environment first. KernelScope requires a CUDA-enabled PyTorch installation. If FrameTrace is elsewhere, set `KERNELSCOPE_FRAMETRACE_DIR` to its absolute directory. The original combined-workspace layout remains supported.

Use Python 3.11+ and a CUDA-enabled PyTorch/torchvision installation on Windows or Ubuntu. Prepare the shared assets first if absent:

```sh
python projects/perception-replay/run.py prepare-real
python projects/perception-replay/run.py demo --real
python projects/kernel-scope/run.py demo --iterations 10 --repeats 3 --warmup 5 --batch-sizes 1 4
python projects/kernel-scope/profile_run.py
```

The report retains raw CUDA-event timings, synchronized wall-clock trial medians, randomized configuration order, peak PyTorch allocated memory, input and checkpoint hashes, and per-trial detection comparisons. Profiling is a separate instrumented run; see PROFILING.md.

## Correctness and limits

Compare all-class detections above confidence 0.3 with the same-batch FP32 reference. Match greedily by class and IoU at least 0.9. The gate requires finite raw boxes/scores, a nonempty reference, and preservation F1 at least 0.95 in every trial. This checks output preservation, not accuracy against human labels. Thresholds are explicit diagnostic choices, not validated deployment requirements.

Frames come from one correlated video. Inputs reside on the GPU before timing; detection transforms and postprocessing are included, but image decoding, transfer and loading are excluded. CUDA-event intervals can contain CPU launch gaps. Memory is peak PyTorch allocation, not all process VRAM. Laptop thermals, power settings and desktop applications affect measurements. No universal speedup, custom CUDA kernel, cloud savings, or production service performance is claimed. Compilation/TensorRT and independent task-quality evaluation remain future work.

Source assets and weights stay in FrameTrace; see ATTRIBUTION.md. The website displays recorded results and does not execute CUDA in a browser.

Timing follows [PyTorch CUDA guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Repository

Source: https://github.com/sanathhiremath/kernelscope


This repository contains a local working prototype. Reproduction commands, measured scope, and limitations are documented above.
