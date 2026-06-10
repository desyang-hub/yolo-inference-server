#!/usr/bin/env python3
"""
推理服务基准测试客户端

用于测试推理服务的性能和并发能力。

使用示例:
    # 基本测试
    python benchmark_client.py -i test.jpg

    # 并发测试
    python benchmark_client.py -i test.jpg -n 100 -c 10

    # 指定服务器地址
    python benchmark_client.py -i test.jpg -u http://localhost:8080
"""

import argparse
import io
import json
import time
import concurrent.futures
import sys

try:
    import requests
except ImportError:
    print("Error: requests library not installed.")
    print("Install with: pip install requests")
    sys.exit(1)


def send_request(server_url, image_path):
    """发送单个推理请求"""
    url = f"{server_url}/infer"

    with open(image_path, "rb") as f:
        image_data = f.read()

    t0 = time.time()
    response = requests.post(url, files={"image": image_data}, timeout=30)
    latency = time.time() - t0

    return {
        "latency": latency,
        "status_code": response.status_code,
        "response": response.json() if response.status_code == 200 else None,
    }


def run_benchmark(server_url, image_path, num_requests, concurrency):
    """运行基准测试"""
    print(f"\n{'='*60}")
    print(f"  Inference Server Benchmark")
    print(f"{'='*60}")
    print(f"Server URL:    {server_url}")
    print(f"Image:         {image_path}")
    print(f"Requests:      {num_requests}")
    print(f"Concurrency:   {concurrency}")
    print(f"{'='*60}\n")

    # 发送请求
    start_time = time.time()
    latencies = []
    success_count = 0
    error_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 提交所有请求
        futures = [
            executor.submit(send_request, server_url, image_path)
            for _ in range(num_requests)
        ]

        # 收集结果
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                latencies.append(result["latency"])

                if result["status_code"] == 200:
                    success_count += 1
                else:
                    error_count += 1

                # 进度显示
                if (i + 1) % max(1, num_requests // 10) == 0:
                    print(f"\rProgress: {i + 1}/{num_requests}", end="")

            except Exception as e:
                error_count += 1
                print(f"\nError: {e}")

    total_time = time.time() - start_time

    # 计算统计
    latencies.sort()
    qps = num_requests / total_time if total_time > 0 else 0

    p50 = latencies[len(latencies) // 2] * 1000
    p90 = latencies[int(len(latencies) * 0.9)] * 1000
    p99 = latencies[int(len(latencies) * 0.99)] * 1000
    avg = sum(latencies) / len(latencies) * 1000
    min_lat = latencies[0] * 1000
    max_lat = latencies[-1] * 1000

    # 输出结果
    print(f"\n\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"Total Time:    {total_time:.2f}s")
    print(f"QPS:           {qps:.2f}")
    print(f"Success:       {success_count}/{num_requests}")
    if error_count > 0:
        print(f"Errors:        {error_count}")
    print(f"")
    print(f"Latency (ms):")
    print(f"  Min:           {min_lat:.2f}")
    print(f"  Avg:           {avg:.2f}")
    print(f"  P50:           {p50:.2f}")
    print(f"  P90:           {p90:.2f}")
    print(f"  P99:           {p99:.2f}")
    print(f"  Max:           {max_lat:.2f}")
    print(f"{'='*60}\n")

    return {
        "qps": qps,
        "latency_avg": avg,
        "latency_p50": p50,
        "latency_p90": p90,
        "latency_p99": p99,
        "success_rate": success_count / num_requests,
    }


def test_health(server_url):
    """测试健康检查端点"""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Health Check: OK ({data.get('status', 'unknown')})")
            return True
        else:
            print(f"Health Check: Failed (status {response.status_code})")
            return False
    except Exception as e:
        print(f"Health Check: Error ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Inference Server Benchmark")
    parser.add_argument(
        "-u", "--url", default="http://localhost:8080", help="Server URL"
    )
    parser.add_argument("-i", "--image", required=True, help="Test image path")
    parser.add_argument(
        "-n", "--num-requests", type=int, default=100, help="Number of requests"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=1, help="Concurrent requests"
    )
    parser.add_argument(
        "--health-only", action="store_true", help="Only check health"
    )

    args = parser.parse_args()

    # 健康检查
    if not test_health(args.url):
        print("Server is not healthy. Exiting.")
        return

    if args.health_only:
        return

    # 运行基准测试
    run_benchmark(args.url, args.image, args.num_requests, args.concurrency)


if __name__ == "__main__":
    main()
