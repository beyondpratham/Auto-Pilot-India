import argparse
import json
import os
import random
import shutil
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

from autopilot.config import CLASS_NAMES, COCO_TO_RISK_CLASS, IMAGE_SIZE, RANDOM_STATE

PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
POTHOLE_URL = (
    "https://github.com/jaygala24/pothole-detection/releases/download/"
    "v1.0.0/Pothole.Dataset.IVCNZ.zip"
)
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{file_name}"

CIFAR_VEHICLE_LABELS = (1, 9)
CIFAR_ANIMAL_LABELS = (3, 4, 5, 7)


def download(url, dest_path, attempts=5, timeout=30):
    if os.path.exists(dest_path):
        print(f"Already downloaded: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    last_error = None
    for attempt in range(1, attempts + 1):
        print(f"Downloading {url} (attempt {attempt}/{attempts})")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            os.replace(tmp_path, dest_path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def extract(zip_path, dest_dir):
    if os.path.isdir(dest_dir) and os.listdir(dest_dir):
        print(f"Already extracted: {dest_dir}")
        return
    print(f"Extracting {zip_path}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)


def extract_member(zip_path, member_name, dest_path):
    if os.path.exists(dest_path):
        print(f"Already extracted: {dest_path}")
        return
    print(f"Extracting {member_name} from {zip_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive, archive.open(member_name) as src, open(dest_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def find_files(root, suffix):
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(suffix):
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    return intersection / area_a if area_a > 0 else 0.0


def process_penn_fudan(extract_dir, human_dir, nothing_dir, max_human, max_nothing, rng):
    mask_paths = find_files(extract_dir, "_mask.png")
    human_count = 0
    nothing_count = 0

    for mask_path in mask_paths:
        if human_count >= max_human and nothing_count >= max_nothing:
            break

        image_path = mask_path.replace("PedMasks", "PNGImages").replace("_mask.png", ".png")
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        width, height = image.size

        if human_count < max_human:
            for object_id in (v for v in np.unique(mask) if v != 0):
                if human_count >= max_human:
                    break
                ys, xs = np.where(mask == object_id)
                if len(xs) == 0:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crop = image.crop((x1, y1, x2, y2)).resize(IMAGE_SIZE)
                crop.save(os.path.join(human_dir, f"pennfudan_{human_count:04d}.jpg"))
                human_count += 1

        if nothing_count < max_nothing:
            occupied = mask > 0
            for _ in range(15):
                if nothing_count >= max_nothing:
                    break
                crop_w = rng.randint(width // 6, max(width // 6 + 1, width // 3))
                crop_h = rng.randint(height // 6, max(height // 6 + 1, height // 3))
                if width - crop_w <= 0 or height - crop_h <= 0:
                    continue
                x0 = rng.randint(0, width - crop_w)
                y0 = rng.randint(0, height - crop_h)
                window = occupied[y0 : y0 + crop_h, x0 : x0 + crop_w]
                if window.mean() > 0.05:
                    continue
                crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(IMAGE_SIZE)
                crop.save(os.path.join(nothing_dir, f"pennfudan_bg_{nothing_count:04d}.jpg"))
                nothing_count += 1

    return human_count, nothing_count


def process_pothole(extract_dir, pothole_dir, nothing_dir, max_pothole, max_nothing, rng):
    label_paths = find_files(extract_dir, ".txt")
    rng.shuffle(label_paths)

    pothole_count = 0
    nothing_count = 0

    for label_path in label_paths:
        if pothole_count >= max_pothole and nothing_count >= max_nothing:
            break

        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = label_path[: -len(".txt")] + ext
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError):
            continue
        width, height = image.size

        with open(label_path) as label_file:
            lines = [line.split() for line in label_file.read().strip().splitlines() if line.strip()]

        boxes = []
        for parts in lines:
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts)
            x1 = max(0, (cx - bw / 2) * width)
            y1 = max(0, (cy - bh / 2) * height)
            x2 = min(width, (cx + bw / 2) * width)
            y2 = min(height, (cy + bh / 2) * height)
            boxes.append((x1, y1, x2, y2))

        if not boxes:
            continue

        if pothole_count < max_pothole:
            x1, y1, x2, y2 = boxes[0]
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crop = image.crop((x1, y1, x2, y2)).resize(IMAGE_SIZE)
                crop.save(os.path.join(pothole_dir, f"pothole_{pothole_count:04d}.jpg"))
                pothole_count += 1

        if nothing_count < max_nothing:
            crop_w, crop_h = max(1, width // 4), max(1, height // 4)
            for _ in range(10):
                if width - crop_w <= 0 or height - crop_h <= 0:
                    break
                x0 = rng.randint(0, width - crop_w)
                y0 = rng.randint(0, height - crop_h)
                candidate_box = (x0, y0, x0 + crop_w, y0 + crop_h)
                if any(iou(candidate_box, box) > 0.05 for box in boxes):
                    continue
                crop = image.crop(candidate_box).resize(IMAGE_SIZE)
                crop.save(os.path.join(nothing_dir, f"pothole_bg_{nothing_count:04d}.jpg"))
                nothing_count += 1
                break

    return pothole_count, nothing_count


def process_cifar(cache_dir, vehicle_dir, animal_dir, vehicle_per_class, animal_per_class, rng):
    dataset = CIFAR10(root=cache_dir, train=True, download=True)
    data = dataset.data
    targets = np.array(dataset.targets)

    def save_class(label, out_dir, count, per_class):
        indices = list(np.where(targets == label)[0])
        rng.shuffle(indices)
        for idx in indices[:per_class]:
            image = Image.fromarray(data[idx]).resize(IMAGE_SIZE, Image.BICUBIC)
            image.save(os.path.join(out_dir, f"cifar_{label}_{count:04d}.jpg"))
            count += 1
        return count

    vehicle_count = 0
    for label in CIFAR_VEHICLE_LABELS:
        vehicle_count = save_class(label, vehicle_dir, vehicle_count, vehicle_per_class)

    animal_count = 0
    for label in CIFAR_ANIMAL_LABELS:
        animal_count = save_class(label, animal_dir, animal_count, animal_per_class)

    return vehicle_count, animal_count


def fetch_image(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def process_coco(annotations_path, image_cache_dir, class_dirs, remaining, rng, min_box_size=40):
    with open(annotations_path) as ann_file:
        coco = json.load(ann_file)

    id_to_filename = {image["id"]: image["file_name"] for image in coco["images"]}
    category_names = {category["id"]: category["name"] for category in coco["categories"]}

    boxes_by_class = {"human": [], "vehicle": [], "animal": []}
    boxes_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        if w < min_box_size or h < min_box_size:
            continue
        box = (x, y, x + w, y + h)
        boxes_by_image.setdefault(ann["image_id"], []).append(box)

        risk_class = COCO_TO_RISK_CLASS.get(category_names.get(ann["category_id"]))
        if risk_class in boxes_by_class:
            boxes_by_class[risk_class].append((ann["image_id"], box))

    os.makedirs(image_cache_dir, exist_ok=True)
    image_cache = {}

    def get_image(image_id):
        if image_id in image_cache:
            return image_cache[image_id]

        file_name = id_to_filename.get(image_id)
        if file_name is None:
            image_cache[image_id] = None
            return None

        local_path = os.path.join(image_cache_dir, file_name)
        try:
            if not os.path.exists(local_path):
                data = fetch_image(COCO_IMAGE_URL_TEMPLATE.format(file_name=file_name))
                with open(local_path, "wb") as out_file:
                    out_file.write(data)
            image = Image.open(local_path).convert("RGB")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            image_cache[image_id] = None
            return None

        image_cache[image_id] = image
        return image

    counts = {"human": 0, "vehicle": 0, "animal": 0}
    used_image_ids = set()

    for class_name, items in boxes_by_class.items():
        target = remaining.get(class_name, 0)
        if target <= 0:
            continue
        rng.shuffle(items)
        for image_id, box in items:
            if counts[class_name] >= target:
                break
            image = get_image(image_id)
            if image is None:
                continue
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = image.crop((x1, y1, x2, y2)).resize(IMAGE_SIZE)
            crop.save(os.path.join(class_dirs[class_name], f"coco_{class_name}_{counts[class_name]:04d}.jpg"))
            counts[class_name] += 1
            used_image_ids.add(image_id)

    nothing_target = remaining.get("nothing", 0)
    nothing_count = 0
    if nothing_target > 0:
        candidate_ids = list(used_image_ids)
        rng.shuffle(candidate_ids)
        for image_id in candidate_ids:
            if nothing_count >= nothing_target:
                break
            image = image_cache.get(image_id)
            if image is None:
                continue
            boxes = boxes_by_image.get(image_id, [])
            width, height = image.size
            crop_w, crop_h = max(1, width // 4), max(1, height // 4)
            for _ in range(8):
                if width - crop_w <= 0 or height - crop_h <= 0:
                    break
                x0 = rng.randint(0, width - crop_w)
                y0 = rng.randint(0, height - crop_h)
                candidate_box = (x0, y0, x0 + crop_w, y0 + crop_h)
                if any(iou(candidate_box, box) > 0.05 for box in boxes):
                    continue
                crop = image.crop(candidate_box).resize(IMAGE_SIZE)
                crop.save(os.path.join(class_dirs["nothing"], f"coco_bg_{nothing_count:04d}.jpg"))
                nothing_count += 1
                break

    return counts, nothing_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and assemble a real image dataset for the classification benchmark"
    )
    parser.add_argument("--output", default="data/real")
    parser.add_argument("--cache", default="data/.cache")
    parser.add_argument("--human-count", type=int, default=250)
    parser.add_argument("--vehicle-per-class", type=int, default=75)
    parser.add_argument("--animal-per-class", type=int, default=40)
    parser.add_argument("--pothole-count", type=int, default=200)
    parser.add_argument("--nothing-count", type=int, default=200)
    parser.add_argument("--human-total", type=int, default=350)
    parser.add_argument("--vehicle-total", type=int, default=350)
    parser.add_argument("--animal-total", type=int, default=350)
    parser.add_argument("--nothing-total", type=int, default=300)
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def count_files(directory):
    return len(os.listdir(directory))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    class_dirs = {name: os.path.join(args.output, name) for name in CLASS_NAMES}
    for class_dir in class_dirs.values():
        if os.path.isdir(class_dir):
            shutil.rmtree(class_dir)
        os.makedirs(class_dir, exist_ok=True)

    penn_zip = os.path.join(args.cache, "PennFudanPed.zip")
    penn_dir = os.path.join(args.cache, "PennFudanPed")
    download(PENN_FUDAN_URL, penn_zip)
    extract(penn_zip, penn_dir)
    human_count, nothing_from_penn = process_penn_fudan(
        penn_dir, class_dirs["human"], class_dirs["nothing"],
        args.human_count, args.nothing_count // 2, rng,
    )
    print(f"Penn-Fudan Pedestrians: {human_count} human crops, {nothing_from_penn} background crops")

    pothole_zip = os.path.join(args.cache, "pothole.zip")
    pothole_dir = os.path.join(args.cache, "pothole_raw")
    download(POTHOLE_URL, pothole_zip)
    extract(pothole_zip, pothole_dir)
    pothole_count, nothing_from_pothole = process_pothole(
        pothole_dir, class_dirs["pothole"], class_dirs["nothing"],
        args.pothole_count, args.nothing_count - nothing_from_penn, rng,
    )
    print(f"Pothole dataset (jaygala24/pothole-detection): {pothole_count} pothole crops, "
          f"{nothing_from_pothole} background crops")

    vehicle_count, animal_count = process_cifar(
        os.path.join(args.cache, "cifar10"), class_dirs["vehicle"], class_dirs["animal"],
        args.vehicle_per_class, args.animal_per_class, rng,
    )
    print(f"CIFAR-10: {vehicle_count} vehicle crops, {animal_count} animal crops")

    if not args.skip_coco:
        coco_zip = os.path.join(args.cache, "coco_annotations.zip")
        coco_json = os.path.join(args.cache, "instances_val2017.json")
        coco_image_cache = os.path.join(args.cache, "coco_images")
        download(COCO_ANNOTATIONS_URL, coco_zip)
        extract_member(coco_zip, "annotations/instances_val2017.json", coco_json)

        remaining = {
            "human": max(0, args.human_total - count_files(class_dirs["human"])),
            "vehicle": max(0, args.vehicle_total - count_files(class_dirs["vehicle"])),
            "animal": max(0, args.animal_total - count_files(class_dirs["animal"])),
            "nothing": max(0, args.nothing_total - count_files(class_dirs["nothing"])),
        }
        print(f"Fetching from COCO to reach targets: {remaining}")
        coco_counts, coco_nothing = process_coco(
            coco_json, coco_image_cache, class_dirs, remaining, rng
        )
        print(f"COCO 2017: {coco_counts}, {coco_nothing} background crops")

    print("\nFinal dataset:")
    for name, class_dir in class_dirs.items():
        print(f"  {name:<10} {len(os.listdir(class_dir))} images")


if __name__ == "__main__":
    main()
