# Auto-Pilot India

Object detection and risk-weighted navigation decisions for autonomous vehicles in
Indian road conditions, built for CSE343.

The project has two parts:

1. **Navigation pipeline** (`scripts/run_navigation.py`) — a real, pretrained Faster
   R-CNN object detector (COCO weights, via torchvision) finds people/vehicles/animals
   in a frame with real bounding boxes and confidence scores. The forward view is
   split into three segments (left/straight/right); each detection's risk contributes
   to whichever segment it falls in, and the vehicle picks the lowest-risk segment, or
   brakes if every segment is risky. This needs no training and no dataset — it runs
   on any image, video, or webcam feed out of the box.

2. **Classification benchmark** (`scripts/train_classifiers.py`) — the CSE343
   coursework comparison from the project report: HOG+SVM, Logistic Regression,
   Gaussian Naive Bayes, Random Forest, and a transfer-learning CNN, trained and
   evaluated on a real, downloaded dataset (see [data/README.md](data/README.md)),
   with an accuracy/precision/recall/F1 comparison table and chart.

Original project reports, proposal, and slides are kept in [resources/](resources/).

## Project layout

```
autopilot/            core package
  config.py           class names, risk weights, COCO->risk mapping, thresholds
  detector.py          pretrained Faster R-CNN wrapper (real bounding-box detection)
  navigation.py         detection-based left/straight/right/brake decision logic
  datasets.py           class-folder image dataset loading + train/test split
  features.py           HOG feature extraction (OpenCV)
  classical_models.py   HOG+SVM / Logistic Regression / Gaussian NB / Random Forest
  cnn_model.py           transfer-learning CNN (PyTorch, VGG16/MobileNetV2 backbone)
  metrics.py             accuracy/precision/recall/F1 + comparison chart
scripts/
  run_navigation.py           run the detector + driving decision on an image/video/webcam
  download_real_dataset.py     download & assemble a real dataset for the benchmark
  generate_demo_dataset.py      synthetic fallback dataset for an offline sanity check
  train_classifiers.py          train + compare all 5 benchmark models
tests/                 pytest unit tests (no dataset/network required)
data/                  expected dataset location (see data/README.md)
resources/             project reports, proposal, and slide deck (PDF/PPTX)
```

## Setup

Everything runs inside a local virtual environment — nothing is installed globally.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

On bash/WSL/macOS/Linux, activate with `source .venv/bin/activate` instead.

> **Note on frameworks:** the original project sketch used TensorFlow/Keras. This
> Python environment (3.14) has no TensorFlow wheel available yet, so the CNN and the
> object detector are implemented with PyTorch + torchvision instead.
> `requirements.txt` points at PyTorch's CPU-only wheel index, so no CUDA toolkit is
> required — everything here runs on CPU, just slower than with a GPU.
> `opencv-python-headless` and `scikit-learn` are pinned to specific (slightly older)
> versions because their newest releases either dropped an API this project uses
> (`cv2.HOGDescriptor`, removed in OpenCV 5.0) or repeatedly tripped a Windows Smart
> App Control block on a brand-new, not-yet-reputable compiled extension
> (`opencv-python-headless` 4.14's `cv2` native module, scikit-learn 1.9's
> `_gradient_boosting`). If you ever see `ImportError: DLL load failed ... Application
> Control policy has blocked this file` on a *different* file after upgrading a
> dependency, it's almost always the same cause — pin that package back a version or
> two rather than disabling Smart App Control.

## Usage

Run everything from the repository root with the venv activated.

### Navigation pipeline (real detector, no training needed)

```
python scripts/run_navigation.py --image path/to/frame.jpg
```

The first run downloads the pretrained detector weights (~74MB) automatically. Add
`--video path/to/clip.mp4` or `--camera 0` for a video file or webcam instead of a
single image; the annotated result (bounding boxes, segment lines, decision) is
written to `--output` (default `outputs/navigation_result.png`). Lower
`--score-threshold` (default 0.5) to pick up smaller/more distant objects in dense
scenes, at the cost of more false positives.

### Classification benchmark

**1. Get a real dataset:**

```
python scripts/download_real_dataset.py --output data/real
```

Downloads and assembles real photographs from public, no-login sources (Penn-Fudan
Pedestrians, a pothole detection dataset, CIFAR-10, and COCO 2017) — see
[data/README.md](data/README.md) for exactly what comes from where. A synthetic,
network-free fallback is also available via
`python scripts/generate_demo_dataset.py --output data/demo` if you just want to
smoke-test the pipeline.

**2. Train and compare models:**

```
python scripts/train_classifiers.py --data data/real --epochs 15
```

This trains HOG+SVM, Logistic Regression, Gaussian Naive Bayes, Random Forest, and the
CNN, prints an accuracy/precision/recall/F1 comparison table, saves a comparison chart
to `outputs/model_comparison.png`, and saves the best-performing model to `outputs/`.

On the assembled real dataset (1,550 images: 350 human, 350 vehicle, 350 animal, 300
nothing, 200 pothole — see [data/README.md](data/README.md) for sourcing), a reference
run with the MobileNetV2 CNN backbone gave:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| HOG + SVM | 0.645 | 0.643 | 0.645 | 0.644 |
| Logistic Regression | 0.665 | 0.663 | 0.665 | 0.662 |
| Gaussian Naive Bayes | 0.610 | 0.606 | 0.610 | 0.589 |
| Random Forest | 0.648 | 0.644 | 0.648 | 0.644 |
| CNN | 0.897 | 0.898 | 0.897 | 0.896 |

Same relative ordering as the original report (CNN best, Naive Bayes weakest), on real
photographs with real evaluation instead of `np.random.uniform(...)`. Note these
numbers are *lower* than an earlier CIFAR-only version of this dataset (which scored
96%+ on the CNN) — that's expected, not a regression: mixing in COCO's diverse,
in-the-wild photos (varied scale, pose, occlusion, background) makes the benchmark
harder and more representative of real deployment conditions, at the cost of the
inflated scores a narrower, more uniform dataset produces.

## Tests

```
pytest
```

Tests cover feature extraction and the navigation decision logic with fixed,
hand-constructed detections — they do not require a dataset or network access.

## Design notes on what changed from the original sketch

The original notebook export was not runnable: it referenced undefined variables and
non-existent APIs (`load_dataset`, a global `image_dir`), it tried to read object
bounding boxes out of a plain image classifier (which cannot produce them), and its
final "results" were literally `np.random.uniform(...)` calls rather than real
evaluation. This rewrite keeps the modeling ideas from the reports (HOG+SVM, Logistic
Regression, Gaussian Naive Bayes, Random Forest, CNN; a segment-based
left/straight/right decision; the reports' stated intent to use Faster R-CNN) but
makes them actually work, on real data:

- Classification models are benchmarked on real HOG/CNN features extracted from real
  photographs, with metrics computed from actual predictions on a held-out test split.
- The steering decision no longer asks a classifier for bounding boxes — instead it
  runs a real pretrained Faster R-CNN detector and computes segment risk from actual
  detected boxes, their confidence, and a proximity estimate from box size. This
  mirrors the report's *stated* plan to use Faster R-CNN, which the original code
  never actually implemented.
- A `Brake` outcome was added for when every segment is high-risk — this was called
  out as a missing feature in the final report's own conclusions.

## Known limitations

- `pothole` has no equivalent class in COCO (what the pretrained detector knows), so
  it only appears in the classification benchmark, not in the live navigation
  detector. A real deployment would need a dedicated pothole detector/dataset — and
  it's the one class that could not be topped up with a second real source, so it
  stays capped around 200 images.
- Roughly half of `vehicle`/`animal` still comes from CIFAR-10, whose images are
  natively 32x32 pixels (upscaled for training) — lower detail than the COCO and
  Penn-Fudan crops mixed in alongside them.
- The risk-weighting scheme (fixed per-category weights, a linear proximity term from
  box height, brake if the safest segment still exceeds a fixed threshold) is a
  simple, explainable baseline, not a calibrated safety system — treat it as a
  demonstration of the decision logic, not a production ADAS controller.
- Faster R-CNN on CPU runs at roughly 1-2 frames/second on typical images; real-time
  video needs a GPU or a lighter/quantized detector.
