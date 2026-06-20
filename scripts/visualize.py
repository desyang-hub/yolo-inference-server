#!/usr/bin/env python3
"""
推理结果可视化工具

向推理服务器发送图像/视频帧，将检测结果可视化后保存或实时显示。

支持的数据源:
  - 单张图片  (-i / --image)
  - 目录批量图片  (-d / --dir)
  - 视频文件  (-v / --video)
  - 实时摄像头  (-c / --camera)

使用示例:
    # 单图推理 + 可视化
    python scripts/visualize.py -i assets/bus.jpg

    # 批量处理 assets 下所有图片
    python scripts/visualize.py --dir assets/

    # 视频推理，输出带检测框的视频
    python scripts/visualize.py -v recording.mp4

    # 摄像头实时推理，窗口显示
    python scripts/visualize.py -c

    # 指定输出目录和置信度阈值
    python scripts/visualize.py -i assets/bus.jpg -o results/ -t 0.3

    # 指定服务器地址
    python scripts/visualize.py -i assets/bus.jpg -u http://localhost:9000
"""

import argparse
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

        # 标签文字（同色，无背景）
        label = f"{cls_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6 * thickness, thickness)
        # 文字放在框上方，如果超出图像顶部则放在框内
        text_y = max(y1 - 4, th + 4)
        cv2.putText(vis, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thickness,
                    color, thickness)

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


def infer_frame(server_url: str, frame: np.ndarray, conf_threshold: float) -> tuple:
    """
    对单帧图像发送推理请求并绘制检测结果。

    Returns:
        (vis_image, detection_count, latency_ms) 或 (None, 0, -1) 失败时
    """
    t0 = time.time()

    # 将帧编码为 JPEG 字节
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    byte_data = buf.tobytes()

    try:
        resp = requests.post(
            f"{server_url}/infer",
            files={"image": ("frame.jpg", byte_data, "image/jpeg")},
            timeout=60,
        )
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")
        return None, 0, -1

    latency_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        return None, 0, -1

    result = resp.json()
    if result.get("status") != "ok":
        return None, 0, -1

    detections = result.get("detections", [])
    vis = draw_detections(frame, detections, conf_threshold)
    count = len([d for d in detections if d["confidence"] >= conf_threshold])

    # 在左上角叠加帧统计信息
    stats_text = [
        f"FPS: {1.0 / max(latency_ms / 1000, 0.001):.1f}",
        f"Latency: {latency_ms:.0f}ms",
        f"Detections: {count}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    for i, line in enumerate(stats_text):
        y = margin + 16 + i * 18
        cv2.putText(vis, line, (margin, y), font, 0.5,
                    (0, 255, 0), 1)

    return vis, count, latency_ms


def infer_video(server_url: str, video_path: str,
                conf_threshold: float, output_dir: str,
                skip_frames: int = 1) -> dict:
    """
    对视频文件逐帧推理，输出带检测结果的视频。

    Args:
        server_url:       推理服务器地址
        video_path:       输入视频路径
        conf_threshold:   置信度阈值
        output_dir:       输出目录
        skip_frames:      跳帧数（每 (skip_frames+1) 帧推理一次以加速）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Failed to open video: {video_path}")
        return {"status": "error", "error": "video open failed"}

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {w}x{h}, FPS={fps:.1f}, Total frames={total_frames}")

    # 输出视频路径
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{stem}_result.mp4")
    idx = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{stem}_result_{idx}.mp4")
        idx += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"  Output: {out_path}")
    print(f"  Skip frames: {skip_frames} (inference every {skip_frames + 1} frames)")
    print()

    # 推理统计
    frame_idx = 0
    inferred_frames = 0
    total_detections = 0
    latencies = []

    # 缓存上一帧的检测结果，用于跳过帧的绘制
    last_vis = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (skip_frames + 1) == 0:
                # 需要推理的帧
                vis, count, latency = infer_frame(
                    server_url, frame, conf_threshold
                )
                if vis is not None:
                    last_vis = vis
                    total_detections += count
                    latencies.append(latency)
                    inferred_frames += 1
                else:
                    # 推理失败，使用原始帧
                    last_vis = frame
            else:
                # 跳过的帧，直接复制原始帧
                last_vis = frame

            writer.write(last_vis)

            # 进度打印
            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                pct = frame_idx / total_frames * 100
                print(f"  Progress: {frame_idx}/{total_frames} ({pct:.1f}%)")

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted by user.")
    finally:
        cap.release()
        writer.release()

    # 汇总
    avg_lat = np.mean(latencies) if latencies else 0
    print(f"\n{'='*50}")
    print(f"  Video Inference Summary")
    print(f"{'='*50}")
    print(f"  Frames processed:  {frame_idx}")
    print(f"  Frames inferred:   {inferred_frames}")
    print(f"  Total detections:  {total_detections}")
    print(f"  Avg latency:       {avg_lat:.1f} ms")
    print(f"  Output:            {out_path}")
    print(f"{'='*50}\n")

    return {
        "status": "ok",
        "frames": frame_idx,
        "inferred": inferred_frames,
        "total_detections": total_detections,
        "avg_latency_ms": avg_lat,
        "output": out_path,
    }


def infer_camera(server_url: str, camera_id: int,
                 conf_threshold: float, skip_frames: int = 1) -> dict:
    """
    打开摄像头进行实时推理，窗口显示检测结果。

    Args:
        server_url:     推理服务器地址
        camera_id:      摄像头设备 ID（0 表示默认摄像头）
        conf_threshold: 置信度阈值
        skip_frames:    跳帧数（每 (skip_frames+1) 帧推理一次）

    按 'q' 或 ESC 退出。
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"  [ERROR] Failed to open camera {camera_id}")
        return {"status": "error", "error": "camera open failed"}

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {actual_w}x{actual_h}, Device ID={camera_id}")
    print(f"  Skip frames: {skip_frames} (inference every {skip_frames + 1} frames)")
    print(f"  Press 'q' or ESC to exit\n")

    # 统计
    frame_idx = 0
    inferred_frames = 0
    total_detections = 0
    latencies = []

    # FPS 计算
    fps_history = []

    # 缓存最新推理结果
    last_vis = None
    last_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  [WARN] Failed to read frame from camera.")
                break

            if frame_idx % (skip_frames + 1) == 0:
                # 需要推理
                frame_start = time.time()
                vis, count, latency = infer_frame(
                    server_url, frame, conf_threshold
                )
                if vis is not None:
                    last_vis = vis
                    last_count = count
                    total_detections += count
                    latencies.append(latency)
                    inferred_frames += 1
                else:
                    last_vis = frame

                # 计算端到端 FPS
                elapsed = (time.time() - frame_start) * 1000
                fps_history.append(elapsed)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            else:
                last_vis = frame

            cv2.imshow("YOLO Detection - Press 'q' to exit", last_vis)

            frame_idx += 1

            # 按 q 或 ESC 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 汇总
    avg_frame_time = np.mean(fps_history) if fps_history else 0
    avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
    avg_lat = np.mean(latencies) if latencies else 0

    print(f"\n{'='*50}")
    print(f"  Camera Inference Summary")
    print(f"{'='*50}")
    print(f"  Frames processed:  {frame_idx}")
    print(f"  Frames inferred:   {inferred_frames}")
    print(f"  Total detections:  {total_detections}")
    print(f"  Avg FPS:           {avg_fps:.1f}")
    print(f"  Avg latency:       {avg_lat:.1f} ms")
    print(f"{'='*50}\n")

    return {
        "status": "ok",
        "frames": frame_idx,
        "inferred": inferred_frames,
        "total_detections": total_detections,
        "avg_fps": avg_fps,
        "avg_latency_ms": avg_lat,
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
  %(prog)s -v recording.mp4                       # video inference
  %(prog)s -c                                     # live camera
  %(prog)s -i assets/bus.jpg -t 0.1 -o output/    # low threshold + custom output
        """,
    )
    parser.add_argument("-u", "--url", default="http://localhost:8080",
                        help="Server URL (default: http://localhost:8080)")
    parser.add_argument("-i", "--image",
                        help="Single image path")
    parser.add_argument("-d", "--dir",
                        help="Directory to process all images in")
    parser.add_argument("-v", "--video",
                        help="Video file path for inference")
    parser.add_argument("-c", "--camera", nargs="?", const=0, default=-1, type=int,
                        help="Live camera inference. Optional device ID (default: 0)")
    parser.add_argument("-o", "--output", default="output/visualize",
                        help="Output directory (default: output/visualize)")
    parser.add_argument("-t", "--threshold", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("-s", "--skip-frames", type=int, default=1,
                        help="Skip N frames between inference (video/camera only, default: 1)")

    args = parser.parse_args()

    # 检查是否有有效数据源
    has_data_source = bool(args.image or args.dir or args.video)
    is_camera = args.camera >= 0

    if not has_data_source and not is_camera:
        parser.print_help()
        sys.exit(1)

    # 健康检查
    print(f"\nServer: {args.url}")
    if not check_health(args.url):
        print("Server is not healthy. Please start the inference server first.")
        sys.exit(1)

    # --- 视频推理模式 ---
    if args.video:
        print(f"\nMode: Video Inference")
        print(f"Input: {args.video}")
        infer_video(args.url, args.video, args.threshold,
                    args.output, args.skip_frames)
        return

    # --- 摄像头实时推理模式 ---
    if is_camera:
        cam_id = args.camera
        print(f"\nMode: Camera Inference (Device {cam_id})")
        infer_camera(args.url, cam_id, args.threshold, args.skip_frames)
        return

    # --- 图片模式 ---
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
