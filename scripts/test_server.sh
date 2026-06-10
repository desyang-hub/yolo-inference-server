#!/bin/bash
# 推理服务测试脚本
#
# 使用示例:
#   ./scripts/test_server.sh --model models/yolov8n.onnx

set -e

SERVER_URL="http://localhost:8080"
MODEL_PATH=""
IMAGE_PATH=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --port)
            SERVER_URL="http://localhost:$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    echo "Usage: $0 --model <path> [--image <path>] [--port <port>]"
    exit 1
fi

echo "=========================================="
echo "  Inference Server Test"
echo "=========================================="

# 1. 启动服务器
echo ""
echo "1. Starting server..."
./build/inference_server --model "$MODEL_PATH" --port 8080 --log-level debug &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"

# 等待服务器启动
sleep 3

# 2. 健康检查
echo ""
echo "2. Health check..."
HEALTH=$(curl -s "$SERVER_URL/health")
echo "   $HEALTH"

# 3. 推理测试
echo ""
echo "3. Inference test..."
if [ -n "$IMAGE_PATH" ]; then
    RESULT=$(curl -s -X POST "$SERVER_URL/infer" \
        -F "image=@$IMAGE_PATH")
    echo "   $RESULT" | python3 -m json.tool 2>/dev/null || echo "   $RESULT"
else
    echo "   Skipped (no image specified)"
fi

# 4. 性能测试
echo ""
echo "4. Performance test (if benchmark_client.py is available)..."
if command -v python3 &> /dev/null && [ -n "$IMAGE_PATH" ]; then
    python3 scripts/benchmark_client.py \
        --url "$SERVER_URL" \
        --image "$IMAGE_PATH" \
        --num-requests 10 \
        --concurrency 1
fi

# 5. 统计信息
echo ""
echo "5. Server stats..."
STATS=$(curl -s "$SERVER_URL/stats")
echo "   $STATS" | python3 -m json.tool 2>/dev/null || echo "   $STATS"

# 6. 关闭服务器
echo ""
echo "6. Shutting down..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "   Done"

echo ""
echo "=========================================="
echo "  Test completed"
echo "=========================================="
