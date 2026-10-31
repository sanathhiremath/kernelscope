# Workload attribution

KernelScope reuses FrameTrace's resized street-video replay frames locally. Source: [People Walking on a Street](https://www.pexels.com/video/people-walking-on-a-street-5365843/), CityXcape, under the [Pexels license](https://www.pexels.com/license/). See `../perception-replay/ATTRIBUTION.md` for the source project attribution. No original video or model weights are bundled in the website.

The frozen SSDLite320 MobileNetV3 Large detector uses torchvision COCO_V1 weights from https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth. The generated report records the checkpoint hash and input frame hashes. No training or fine-tuning is claimed. Model-output preservation is distinct from correctness against human annotations.
