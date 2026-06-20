#!/usr/bin/env python3
"""
推理结果可视化工具

向推理服务器发送单张图像，将检测结果可视化后保存。
也支持对 assets/ 目录下所有图像进行批量推理和可视化。

使用示例:
    # 单图推理 + 可视化
    python scripts/visualize.py -i assets/bus.jpg

    # 批量处理 assets 下所有图片
    python scripts/visualize.py --dir assets/

    # 指定输出目录和置信度阈值
    python scripts/visualize.py -i assets/bus.jpg -o results/ -t 0.3

    # 指定服务器地址
    python scripts/visualize.py -i assets/bus.jpg -u http://localhost:9000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# COCO 80 类别名称（与 C++ 端 CLASS_NAMES 一致）
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# 为每个类别分配一个固定颜色（HSL → RGB 手工转换，避免 headless OpenCV 限制）
def _hsl_to_rgb(h: float, s: float, l: float) -> tuple:
    """HSL → RGB, h in [0,360), s/l in [0,1], returns (B,G,R) in [0,255]."""
    h = h / 360.0
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = l - c / 2
    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def _class_color(class_id: int) -> tuple:
    """为给定类别返回一个 BGR 颜色（OpenCV 格式）。"""
    h = (class_id * 137.508) % 360  # 黄金角均匀分布
    return _hsl_to_rgb(h, 0.85, 0.55)


def draw_detections(image: np.ndarray, detections: list,
                    conf_threshold: float) -> np.ndarray:
    """
    在原图上绘制检测结果。

    Args:
        image:          BGR 原始图像 (cv2)
        detections:     服务器返回的 detections 列表
        conf_threshold: 置信度阈值，低于此值不绘制

    Returns:
        绘制好结果的新图像
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    for det in detections:
        conf = det["confidence"]
        if conf < conf_threshold:
            continue

        cls_id = det["class_id"]
        cls_name = det.get("class_name", f"class{cls_id}")
        bbox = det["bbox"]

        # 服务端返回的是归一化中心坐标 (x, y, width, height)
        cx, cy, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # 转回像素级左上角 + 右下角
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 裁剪到图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = _class_color(cls_id)
        thickness = max(2, min(h, w) // 200)

        # 绘制边界框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness * 2)

        # 标签背景
        label = f"{cls_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6 * thickness, thickness)
        cv2.rectangle(vis, (x1, y1 - th - baseline - 4),
                      (x1 + tw, y1), color, thickness * 3)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thickness,
                    (255, 255, 255), thickness)

    return vis


def infer_single(server_url: str, image_path: str,
                 conf_threshold: float, output_dir: str) -> dict:
    """
    对单张图像发送推理请求，可视化结果并保存。

    Returns:
        包含统计信息的字典
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Failed to read image: {image_path}")
        return {"status": "error", "error": "image read failed"}

    # 发送推理请求
    url = f"{server_url}/infer"
    fname = os.path.basename(image_path)

    t0 = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"image": (fname, f.read())}, timeout=60)
    latency_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        print(f"  [ERROR] Server returned {resp.status_code}: {resp.text[:200]}")
        return {"status": "error", "error": f"http {resp.status_code}"}

    result = resp.json()
    if result.get("status") != "ok":
        print(f"  [ERROR] Inference failed: {result.get('error', 'unknown')}")
        return {"status": "error", "error": result.get("error", "")}

    detections = result.get("detections", [])
    timing = result.get("timing", {})

    # 可视化
    vis = draw_detections(img, detections, conf_threshold)

    # 在图像角落叠加统计信息
    stats_text = [
        f"Inference: {timing.get('inference_time_ms', 0):.1f}ms",
        f"Request latency: {latency_ms:.1f}ms",
        f"Detections: {len([d for d in detections if d['confidence'] >= conf_threshold])}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    for i, line in enumerate(stats_text):
        y = margin + 16 + i * 18
        cv2.putText(vis, line, (margin, y), font, 0.5,
                    (0, 255, 0), 1)

    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{stem}_result.jpg")
    # 同名冲突时加序号（bus.jpg 和 bus.png 都会生成 bus_result.jpg）
    idx = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{stem}_result_{idx}.jpg")
        idx += 1
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  [{fname}] {len(detections)} detections -> {out_path}")
    print(f"    Timing: infer={timing.get('inference_time_ms', 0):.1f}ms, "
          f"total={latency_ms:.1f}ms")

    return {
        "status": "ok",
        "image": image_path,
        "output": out_path,
        "detections": len(detections),
        "inference_ms": timing.get("inference_time_ms", 0),
        "latency_ms": latency_ms,
    }


def check_health(server_url: str) -> bool:
    """检查服务器是否健康。"""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Health: {data.get('status', 'unknown')} "
                  f"(v{data.get('version', '?')})")
            return True
    except Exception as e:
        print(f"  Health check error: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Inference Result Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i assets/bus.jpg                      # single image
  %(prog)s --dir assets/                          # all images in assets/
  %(prog)s -i assets/bus.jpg -t 0.1 -o output/    # low threshold + custom output
        """,
    )
    parser.add_argument("-u", "--url", default="http://localhost:8080",
                        help="Server URL (default: http://localhost:8080)")
    parser.add_argument("-i", "--image",
                        help="Single image path")
    parser.add_argument("-d", "--dir",
                        help="Directory to process all images in")
    parser.add_argument("-o", "--output", default="output/visualize",
                        help="Output directory (default: output/visualize)")
    parser.add_argument("-t", "--threshold", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    # 健康检查
    print(f"\nServer: {args.url}")
    if not check_health(args.url):
        print("Server is not healthy. Please start the inference server first.")
        sys.exit(1)

    # 收集图像列表
    image_paths = []
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if args.image:
        image_paths.append(args.image)

    if args.dir:
        for f in sorted(Path(args.dir).iterdir()):
            if f.suffix.lower() in supported_ext:
                image_paths.append(str(f))

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    print(f"Images: {len(image_paths)}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print()

    # 批量处理
    results = []
    for img_path in image_paths:
        r = infer_single(args.url, img_path, args.threshold, args.output)
        results.append(r)

    # 汇总
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    print(f"  Total images:   {len(results)}")
    print(f"  Success:        {len(ok)}")
    print(f"  Failed:         {len(results) - len(ok)}")
    if ok:
        total_det = sum(r["detections"] for r in ok)
        avg_lat = np.mean([r["latency_ms"] for r in ok])
        avg_inf = np.mean([r["inference_ms"] for r in ok])
        print(f"  Total detections: {total_det}")
        print(f"  Avg inference:    {avg_inf:.1f} ms")
        print(f"  Avg latency:      {avg_lat:.1f} ms")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
