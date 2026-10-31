"""Offline pretrained SSDLite workload reusing attributed FrameTrace assets."""
import hashlib
import os
from pathlib import Path
import torch
from PIL import Image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import pil_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_workload(frame_count=8):
    override = os.environ.get('KERNELSCOPE_FRAMETRACE_DIR')
    candidates = ([Path(override)] if override else [
        PROJECT_ROOT.parent / 'frametrace',
        PROJECT_ROOT.parent / 'perception-replay',
    ])
    source = next((path for path in candidates if (path / 'data/ssdlite320.pth').is_file()), candidates[0])
    checkpoint = source / 'data/ssdlite320.pth'
    if not checkpoint.exists():
        raise FileNotFoundError('Clone frametrace beside kernelscope and run its run.py prepare-real, or set KERNELSCOPE_FRAMETRACE_DIR.')
    paths = sorted((source / 'artifacts/real-frames').glob('*.jpg'))[:frame_count]
    if len(paths) != frame_count:
        raise FileNotFoundError('Generate FrameTrace real footage artifacts first.')
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(pil_to_tensor(image.convert('RGB')).float().div(255))
    provenance = dict(model='SSDLite320 MobileNetV3 Large COCO_V1',
                      weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      weights_source='https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',
                      frames=[dict(image=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
                      source_url='https://www.pexels.com/video/people-walking-on-a-street-5365843/',
                      author='CityXcape', license_url='https://www.pexels.com/license/',
                      attribution='projects/perception-replay/ATTRIBUTION.md')
    return model, tensors, provenance


def serialized_predictions(outputs, threshold=.3):
    predictions = []
    for output in outputs:
        boxes = output['boxes'].detach().float().cpu().tolist()
        labels = output['labels'].detach().cpu().tolist()
        scores = output['scores'].detach().float().cpu().tolist()
        predictions.append([dict(box=box, label=label, score=score)
                            for box, label, score in zip(boxes, labels, scores)
                            if score >= threshold])
    return predictions
