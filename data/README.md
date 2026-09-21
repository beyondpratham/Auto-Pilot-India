# Data

Training scripts expect a folder of images grouped one subfolder per class:

```
data/<dataset-name>/
  animal/
  human/
  nothing/
  pothole/
  vehicle/
```

Class folder names must match `autopilot.config.CLASS_NAMES`.

## Real dataset (recommended)

```
python scripts/download_real_dataset.py --output data/real
```

This downloads and assembles a real, non-synthetic dataset from public sources with
no login/API key required, layering four sources so each class ends up balanced at
around 300-350 images:

- **human** — cropped pedestrian instances from the [Penn-Fudan Pedestrian
  dataset](https://www.cis.upenn.edu/~jshi/ped_html/) (University of Pennsylvania,
  one of the datasets the original project report cited), topped up with `person`
  crops from COCO 2017.
- **pothole** — cropped pothole regions from the MIT-licensed
  [jaygala24/pothole-detection](https://github.com/jaygala24/pothole-detection)
  dataset (1,243 annotated road photos, YOLO-format boxes). No further source is
  layered in — pothole has no equivalent category in CIFAR-10 or COCO.
- **vehicle** — CIFAR-10's `automobile`/`truck` classes, topped up with COCO 2017
  `bicycle`/`car`/`motorcycle`/`bus`/`truck`/`train` crops for higher-resolution,
  in-the-wild variety.
- **animal** — CIFAR-10's `cat`/`dog`/`horse`/`deer` classes, topped up with COCO 2017
  `bird`/`cat`/`dog`/`horse`/`sheep`/`cow`/`elephant`/`bear`/`zebra`/`giraffe` crops.
- **nothing** — background crops sampled away from any annotated object, taken from
  the Penn-Fudan, pothole, and COCO source images.

COCO 2017 crops come from `instances_val2017.json` (only that ~26MB annotation file is
extracted from the ~250MB combined annotations archive) plus the specific images that
contain a needed category, fetched individually from
`images.cocodataset.org` — not the full multi-gigabyte COCO image set.

Raw downloads are cached under `data/.cache/` so re-running the script is fast — each
source is skipped if already downloaded/extracted. Note CIFAR-10 images are only
32x32 natively (upscaled to the training resolution); the COCO crops for the same
classes are much higher resolution, which is exactly why they're layered in — see the
top-level README's "Known limitations" section for the remaining gaps.

## Quick offline check (synthetic demo data)

For a fast, network-free sanity check of the pipeline itself:

```
python scripts/generate_demo_dataset.py --output data/demo --samples-per-class 80
```

This draws simple procedural shapes per class. It exists purely to prove the training
pipeline runs correctly — do not use it to draw conclusions about model quality.

## The navigation/detection pipeline needs no dataset at all

`scripts/run_navigation.py` uses a Faster R-CNN detector pretrained on COCO
(auto-downloaded by torchvision on first use) and does not depend on anything in this
`data/` folder — it works directly on any image, video, or webcam feed.
