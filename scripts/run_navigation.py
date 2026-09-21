import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autopilot.config import DETECTION_SCORE_THRESHOLD
from autopilot.detector import ObjectDetector
from autopilot.navigation import decide_direction

BOX_COLOR = (255, 60, 60)
LINE_COLOR = (255, 255, 0)
TEXT_COLOR = (40, 220, 40)
BRAKE_TEXT_COLOR = (255, 40, 40)


def _text_with_backdrop(draw, xy, text, fill, font):
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    pad = max(2, (bottom - top) // 5)
    draw.rectangle([left - pad, top - pad, right + pad, bottom + pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


def annotate(image_rgb, decision):
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    num_segments = len(decision.segments)
    segment_width = width / num_segments
    scale = max(1.0, width / 640)

    label_font = ImageFont.load_default(size=int(13 * scale))
    decision_font = ImageFont.load_default(size=int(17 * scale))

    for i in range(1, num_segments):
        x = int(i * segment_width)
        draw.line([(x, 0), (x, height)], fill=LINE_COLOR, width=max(1, int(2 * scale)))

    for detection in decision.detections:
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=max(1, int(2 * scale)))
        _text_with_backdrop(
            draw, (x1 + 2, max(0, y1 - 18 * scale)), f"{detection.label} {detection.score:.2f}", BOX_COLOR, label_font
        )

    decision_color = BRAKE_TEXT_COLOR if decision.direction == "Brake" else TEXT_COLOR
    _text_with_backdrop(draw, (10, height - 32 * scale), f"Decision: {decision.direction}", decision_color, decision_font)
    return np.array(pil_image)


def run_on_image(detector, image_path, output_path):
    with Image.open(image_path) as img:
        image_rgb = np.array(img.convert("RGB"))

    detections = detector.detect(image_rgb)
    decision = decide_direction(image_rgb.shape, detections)
    annotated = annotate(image_rgb, decision)
    Image.fromarray(annotated).save(output_path)

    print(f"Detections: {len(detections)}")
    for detection in detections:
        print(f"  {detection.label} (score={detection.score:.2f}) box={tuple(round(v) for v in detection.box)}")
    print(f"Decision: {decision.direction}")
    for segment in decision.segments:
        cause = segment.cause or "clear"
        print(f"  {segment.label}: risk={segment.risk:.4f} ({cause})")
    print(f"Saved annotated frame to {output_path}")


def run_on_video(detector, source, output_path, frame_stride):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    last_decision = None
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if last_decision is None or frame_idx % frame_stride == 0:
                detections = detector.detect(frame_rgb)
                last_decision = decide_direction(frame_rgb.shape, detections)
            annotated_rgb = annotate(frame_rgb, last_decision)
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            frame_idx += 1
    finally:
        capture.release()
        writer.release()
    print(f"Saved annotated video to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the navigation decision pipeline")
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--output", default="outputs/navigation_result.png")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=DETECTION_SCORE_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = ObjectDetector(score_threshold=args.score_threshold)

    if args.image:
        run_on_image(detector, args.image, args.output)
    elif args.video:
        run_on_video(detector, args.video, args.output, args.frame_stride)
    elif args.camera is not None:
        run_on_video(detector, args.camera, args.output, args.frame_stride)
    else:
        raise SystemExit("Provide one of --image, --video, or --camera")


if __name__ == "__main__":
    main()
