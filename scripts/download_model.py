#!/usr/bin/env python3
"""
ONNX 模型下载和转换脚本

用于下载 YOLOv8 ONNX 模型或从 PyTorch 模型转换。

使用示例:
    # 下载预转换的 ONNX 模型
    python download_model.py --model yolov8n --output models/

    # 从 Ultralytics 导出
    python download_model.py --export --model yolov8n --output models/
"""

import argparse
import os
import sys


def download_ultralytics_model(model_name, output_dir):
    """使用 Ultralytics 导出 ONNX 模型"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        print("Install with: pip install ultralytics")
        return False

    print(f"Downloading {model_name} from Ultralytics...")
    model = YOLO(f"{model_name}.pt")

    output_path = os.path.join(output_dir, f"{model_name}.onnx")
    print(f"Exporting to ONNX: {output_path}")

    # 导出为 ONNX
    model.export(format="onnx", imgsz=640, simplify=True)

    # 重命名
    exported_path = os.path.join(output_dir, f"{model_name}.onnx")
    print(f"Model exported: {exported_path}")
    return True


def download_from_url(model_name, output_dir):
    """从 GitHub 下载预转换的模型"""
    import urllib.request

    # Ultralytics 官方 ONNX 模型 URL
    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
    model_url = f"{base_url}/{model_name}.onnx"

    output_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"Downloading {model_name}.onnx...")
    print(f"URL: {model_url}")

    try:
        urllib.request.urlretrieve(model_url, output_path)
        print(f"Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Export from Ultralytics:")
        print("  pip install ultralytics")
        print(f"  python download_model.py --export --model {model_name} --output {output_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models")
    parser.add_argument(
        "--model",
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Model name",
    )
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument(
        "--export", action="store_true", help="Export from Ultralytics"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    if args.export:
        success = download_ultralytics_model(args.model, args.output)
    else:
        success = download_from_url(args.model, args.output)

    if success:
        print(f"\nModel ready: {args.output}/{args.model}.onnx")
        print("Start server:")
        print(f"  ./build/inference_server --model {args.output}/{args.model}.onnx")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
