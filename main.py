import argparse
import json
import os
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from atrous import atrous_correlation_rgb
from utils import histogram_stretch, to_uint8_clip

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rgb_image(path, resize_factor=1.0):
    img = Image.open(path).convert("RGB")
    if resize_factor != 1.0:
        new_w = max(1, int(img.width * resize_factor))
        new_h = max(1, int(img.height * resize_factor))
        img = img.resize((new_w, new_h))
    return np.array(img, dtype=np.uint8)


def save_gray(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def save_rgb(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def apply_filter_from_config(img_rgb, config_path):
    config = load_json(config_path)
    result = atrous_correlation_rgb(
        img_rgb,
        kernel=config["kernel"],
        r=config["r"],
        stride=config["stride"],
        activation=config["activation"]
    )
    return to_uint8_clip(result)

def compute_green_score(img_rgb):
    img = img_rgb.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    score = 2.0 * g - r - b
    return histogram_stretch(score)


def threshold_percentile(gray_img, percentile_value):
    thr = np.percentile(gray_img, percentile_value)
    return gray_img >= thr

def binary_dilate(mask, size=3, iterations=1):
    out = mask.copy()
    radius = size // 2
    for _ in range(iterations):
        padded = np.pad(out, radius, mode="constant", constant_values=False)
        new_mask = np.zeros_like(out, dtype=bool)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y:y + size, x:x + size]
                new_mask[y, x] = np.any(region)
        out = new_mask
    return out


def binary_erode(mask, size=3, iterations=1):
    out = mask.copy()
    radius = size // 2
    for _ in range(iterations):
        padded = np.pad(out, radius, mode="constant", constant_values=False)
        new_mask = np.zeros_like(out, dtype=bool)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y:y + size, x:x + size]
                new_mask[y, x] = np.all(region)
        out = new_mask
    return out


def binary_open(mask, size=3, iterations=1):
    return binary_dilate(binary_erode(mask, size=size, iterations=iterations), size=size, iterations=iterations)


def binary_close(mask, size=5, iterations=1):
    return binary_erode(binary_dilate(mask, size=size, iterations=iterations), size=size, iterations=iterations)


def connected_components(mask):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(pixels)
    return components


def component_bbox(pixels):
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)


def draw_components(img_rgb, components):
    pil_img = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(pil_img)
    for idx, comp in enumerate(components, start=1):
        x1, y1, x2, y2 = component_bbox(comp)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), str(idx), fill=(255, 255, 0))
    return np.array(pil_img, dtype=np.uint8)


def keep_components_by_area(components, min_area, max_area):
    filtered = []
    for comp in components:
        area = len(comp)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered.append(comp)
    return filtered


def write_report(path, count_value, components):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Quantidade estimada de pés de goiaba: {count_value}\n\n")
        for idx, comp in enumerate(components, start=1):
            area = len(comp)
            x1, y1, x2, y2 = component_bbox(comp)
            f.write(f"Objeto {idx}: área={area}, bbox=({x1},{y1}) -> ({x2},{y2})\n")


def run_tree_count(input_path, output_dir, config_path):
    cfg = load_json(config_path)
    img = load_rgb_image(input_path, resize_factor=cfg.get("resize_factor", 1.0))
    os.makedirs(output_dir, exist_ok=True)

    if cfg.get("use_gaussian_preprocess", True):
        img_smooth = apply_filter_from_config(img, cfg["gaussian_config"])
    else:
        img_smooth = img.copy()

    green_score = compute_green_score(img_smooth)
    vegetation_mask = threshold_percentile(green_score, cfg.get("green_threshold_percentile", 70))

    refined_mask = binary_open(
        vegetation_mask,
        size=cfg.get("binary_open_size", 3),
        iterations=cfg.get("binary_open_iterations", 1),
    )
    refined_mask = binary_close(
        refined_mask,
        size=cfg.get("binary_close_size", 5),
        iterations=cfg.get("binary_close_iterations", 1),
    )

    comps = connected_components(refined_mask)
    comps = keep_components_by_area(
        comps,
        min_area=cfg.get("min_area", 120),
        max_area=cfg.get("max_area", None),
    )

    count_value = len(comps)
    overlay = draw_components(img_smooth, comps) if cfg.get("draw_boxes", True) else img_smooth.copy()

    save_gray(os.path.join(output_dir, "01_green_score.png"), green_score)
    save_rgb(os.path.join(output_dir, "02_smoothed.png"), img_smooth)
    save_gray(os.path.join(output_dir, "04_final_mask.png"), (refined_mask.astype(np.uint8) * 255))
    save_rgb(os.path.join(output_dir, "05_overlay_count.png"), overlay)
    write_report(os.path.join(output_dir, "06_count_report.txt"), count_value, comps)

    print(f"Contagem estimada: {count_value}")
    print(f"Saídas salvas em: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Conta pés de goiaba em imagem RGB usando suavização, mapa verde e componentes conectados."
    )
    parser.add_argument("-i", "--input", required=True, help="Imagem RGB de entrada.")
    parser.add_argument("-c", "--config", required=True, help="Configuração JSON da contagem.")
    parser.add_argument("-o", "--output_dir", required=True, help="Diretório de saída.")
    args = parser.parse_args()
    run_tree_count(args.input, args.output_dir, args.config)


if __name__ == "__main__":
    main()
