import argparse
import math
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw

from autopilot.config import CLASS_NAMES, IMAGE_SIZE, RANDOM_STATE


def random_background(size, base_color, jitter=12):
    width, height = size
    base = np.array(base_color, dtype=np.int16).reshape(1, 1, 3)
    noise = np.random.randint(-jitter, jitter + 1, size=(height, width, 3))
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def draw_human(size):
    image = random_background(size, (150, 170, 190))
    draw = ImageDraw.Draw(image)
    width, height = size
    cx = random.randint(width // 4, 3 * width // 4)
    scale = random.uniform(0.6, 1.0)
    head_r = int(0.06 * height * scale)
    top = int(0.15 * height)
    color = tuple(random.randint(20, 90) for _ in range(3))

    draw.ellipse(
        [cx - head_r, top, cx + head_r, top + 2 * head_r], fill=color
    )
    body_top = top + 2 * head_r
    body_bottom = int(body_top + 0.45 * height * scale)
    draw.line([cx, body_top, cx, body_bottom], fill=color, width=max(2, int(0.03 * width)))
    leg_len = int(0.3 * height * scale)
    draw.line([cx, body_bottom, cx - head_r, body_bottom + leg_len], fill=color, width=3)
    draw.line([cx, body_bottom, cx + head_r, body_bottom + leg_len], fill=color, width=3)
    arm_y = body_top + int(0.1 * height * scale)
    draw.line([cx, arm_y, cx - int(head_r * 1.8), arm_y + int(0.15 * height)], fill=color, width=3)
    draw.line([cx, arm_y, cx + int(head_r * 1.8), arm_y + int(0.15 * height)], fill=color, width=3)
    return image


def draw_vehicle(size):
    image = random_background(size, (120, 120, 125))
    draw = ImageDraw.Draw(image)
    width, height = size
    body_w = random.randint(int(0.4 * width), int(0.8 * width))
    body_h = random.randint(int(0.18 * height), int(0.3 * height))
    x0 = random.randint(0, max(1, width - body_w))
    y0 = random.randint(int(0.35 * height), int(0.6 * height))
    color = tuple(random.randint(30, 220) for _ in range(3))

    draw.rectangle([x0, y0, x0 + body_w, y0 + body_h], fill=color)
    draw.rectangle(
        [x0 + body_w * 0.2, y0 - body_h * 0.6, x0 + body_w * 0.7, y0],
        fill=color,
    )
    wheel_r = max(3, int(body_h * 0.3))
    for wx in (x0 + wheel_r, x0 + body_w - wheel_r):
        draw.ellipse(
            [wx - wheel_r, y0 + body_h - wheel_r, wx + wheel_r, y0 + body_h + wheel_r],
            fill=(10, 10, 10),
        )
    return image


def draw_animal(size):
    image = random_background(size, (140, 180, 120))
    draw = ImageDraw.Draw(image)
    width, height = size
    body_w = random.randint(int(0.3 * width), int(0.55 * width))
    body_h = random.randint(int(0.15 * height), int(0.25 * height))
    x0 = random.randint(0, max(1, width - body_w))
    y0 = random.randint(int(0.45 * height), int(0.65 * height))
    color = tuple(random.randint(60, 150) for _ in range(3))

    draw.ellipse([x0, y0, x0 + body_w, y0 + body_h], fill=color)
    head_r = int(body_h * 0.6)
    draw.ellipse(
        [x0 - head_r, y0, x0 + head_r * 0.4, y0 + head_r * 1.5], fill=color
    )
    leg_h = int(0.2 * height)
    for lx in (x0 + body_w * 0.2, x0 + body_w * 0.8):
        draw.line([lx, y0 + body_h, lx, y0 + body_h + leg_h], fill=color, width=4)
    return image


def draw_pothole(size):
    image = random_background(size, (95, 95, 95))
    draw = ImageDraw.Draw(image)
    width, height = size
    cx = random.randint(int(0.3 * width), int(0.7 * width))
    cy = random.randint(int(0.4 * height), int(0.7 * height))
    points = []
    num_points = random.randint(7, 11)
    base_r = random.randint(int(0.08 * width), int(0.18 * width))
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        r = base_r * random.uniform(0.7, 1.15)
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=(25, 22, 20))
    return image


def draw_nothing(size):
    base = random.choice([(90, 90, 95), (60, 110, 60), (150, 170, 190)])
    return random_background(size, base, jitter=18)


GENERATORS = {
    "animal": draw_animal,
    "human": draw_human,
    "nothing": draw_nothing,
    "pothole": draw_pothole,
    "vehicle": draw_vehicle,
}


def generate_dataset(output_dir, samples_per_class, image_size, seed):
    random.seed(seed)
    np.random.seed(seed)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        generator = GENERATORS[class_name]
        for i in range(samples_per_class):
            image = generator(image_size)
            image.save(os.path.join(class_dir, f"{class_name}_{i:04d}.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset")
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_dataset(args.output, args.samples_per_class, tuple(args.image_size), args.seed)
    print(f"Generated {args.samples_per_class} images per class under '{args.output}'.")


if __name__ == "__main__":
    main()
