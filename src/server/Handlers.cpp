/**
 * @file Handlers.cpp
 * @brief HTTP 请求处理器实现
 *
 * 实现 HTTP 请求的解析和响应序列化。
 * 这是一个简化版的 HTTP 解析器，仅支持本项目所需的功能。
 */

#include "inference/server/Handlers.hpp"
#include "inference/common/Logger.hpp"
#include "inference/inference/InferenceService.hpp"

#include <sstream>
#include <algorithm>
#include <cstdio>

// nlohmann/json
#include <nlohmann/json.hpp>

// Base64 解码表
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * @brief Base64 解码
 */
static std::string base64_decode(const std::string& encoded) {
    std::string decoded;

    for (size_t i = 0; i < encoded.size(); ++i) {
        char c = encoded[i];
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') {
            continue;
        }

        auto found = base64_chars.find(c);
        if (found == std::string::npos) {
            continue; // 跳过无效字符
        }

        unsigned int triplet = (found << 18);

        if (++i < encoded.size()) {
            c = encoded[i];
            if (c == '=') {
                triplet <<= 6;
            } else {
                found = base64_chars.find(c);
                triplet |= (found << 12);
                if (++i < encoded.size()) {
                    c = encoded[i];
                    if (c == '=') {
                        triplet <<= 6;
                    } else {
                        found = base64_chars.find(c);
                        triplet |= (found << 6);
                        if (++i < encoded.size()) {
                            c = encoded[i];
                            if (c != '=') {
                                found = base64_chars.find(c);
                                triplet |= found;
                            }
                        }
                    }
                    decoded.push_back(static_cast<char>((triplet >> 16) & 0xFF));
                    decoded.push_back(static_cast<char>((triplet >> 8) & 0xFF));
                } else {
                    decoded.push_back(static_cast<char>((triplet >> 16) & 0xFF));
                    decoded.push_back(static_cast<char>((triplet >> 8) & 0xFF));
                }
            }
        } else {
            decoded.push_back(static_cast<char>((triplet >> 16) & 0xFF));
        }
    }

    return decoded;
}

namespace inference {

// ============================================================
// HttpResponse 实现
// ============================================================

std::string HttpResponse::Serialize() const {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    oss << body;
    return oss.str();
}

// ============================================================
// RequestHandlers 实现
// ============================================================

RequestHandlers::RequestHandlers(InferenceService* service)
    : service_(service) {}

void RequestHandlers::OnConnection(const muduo::net::TcpConnectionPtr& conn) {
    if (conn->connected()) {
        LOG_INFO("Connection established from {}", conn->peerAddress().toIpPort());
    } else {
        LOG_INFO("Connection closed from {}", conn->peerAddress().toIpPort());
    }
}

void RequestHandlers::OnMessage(const muduo::net::TcpConnectionPtr& conn,
                                 muduo::net::Buffer* buffer,
                                 muduo::Timestamp) {
    // 读取所有可用数据
    std::string data(buffer->peek(), buffer->readableBytes());
    buffer->retrieveAll();

    // 解析 HTTP 请求
    HttpRequest request = ParseRequest(data);
    if (!request.parsed) {
        LOG_WARNING("Failed to parse HTTP request from {}",
                    conn->peerAddress().toIpPort());
        HttpResponse error = HandleError(400, "Bad Request");
        conn->send(error.Serialize());
        conn->shutdown();
        return;
    }

    LOG_DEBUG("HTTP {} {} from {}",
              request.method, request.path,
              conn->peerAddress().toIpPort());

    // 路由处理
    HttpResponse response;
    if (request.path == "/infer" && request.method == "POST") {
        response = HandleInfer(request);
    } else if (request.path == "/health") {
        response = HandleHealth();
    } else if (request.path == "/stats") {
        response = HandleStats();
    } else {
        response = HandleNotFound();
    }

    // 发送响应
    std::string response_str = response.Serialize();
    conn->send(response_str);
    conn->shutdown(); // 关闭连接（HTTP/1.0 风格，简单可靠）
}

HttpRequest RequestHandlers::ParseRequest(const std::string& data) {
    HttpRequest request;
    request.parsed = false;

    // 查找请求头和主体的分隔符
    auto header_end = data.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        // 也支持 \n\n 分隔
        header_end = data.find("\n\n");
        if (header_end == std::string::npos) {
            return request;
        }
    }

    std::string headers = data.substr(0, header_end);
    request.body = data.substr(header_end + 4); // 跳过 \r\n\r\n

    // 解析请求行
    auto first_line_end = headers.find("\n");
    if (first_line_end == std::string::npos) {
        first_line_end = headers.find("\r\n");
    }
    if (first_line_end == std::string::npos) {
        return request;
    }

    std::string request_line = headers.substr(0, first_line_end);
    std::istringstream iss(request_line);
    iss >> request.method >> request.path;

    // 解析 Content-Type
    auto ct_pos = headers.find("Content-Type:");
    if (ct_pos != std::string::npos) {
        auto line_end = headers.find("\n", ct_pos);
        request.content_type = headers.substr(ct_pos + 13,
                                               line_end - ct_pos - 13);
        // 去除首尾空格
        request.content_type.erase(0, request.content_type.find_first_not_of(" \t"));
        request.content_type.erase(request.content_type.find_last_not_of(" \t\r\n") + 1);
    }

    request.parsed = true;
    return request;
}

cv::Mat RequestHandlers::ExtractImage(const std::string& body,
                                       const std::string& content_type) {
    // 情况 1: 直接是二进制图像数据
    if (content_type.find("image/") != std::string::npos) {
        std::vector<uint8_t> data(body.begin(), body.end());
        return cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
    }

    // 情况 2: multipart/form-data
    if (content_type.find("multipart/") != std::string::npos) {
        // 查找 boundary
        auto boundary_pos = content_type.find("boundary=");
        if (boundary_pos != std::string::npos) {
            std::string boundary = "--" + content_type.substr(boundary_pos + 9);

            // 简化处理：查找 boundary 之间的二进制数据
            auto start = body.find(boundary);
            if (start != std::string::npos) {
                start = body.find("\n", body.find("\n", start) + 1);
                if (start != std::string::npos) {
                    auto end = body.find(boundary, start);
                    if (end == std::string::npos) {
                        end = body.size() - 1;
                    }

                    // 跳过 header 部分
                    auto header_end = body.find("\n\n", start);
                    if (header_end == std::string::npos) {
                        header_end = body.find("\r\n\r\n", start);
                    }
                    if (header_end != std::string::npos) {
                        std::string img_data = body.substr(header_end + 4,
                                                           end - header_end - 4);
                        // 去除可能的 \r\n
                        while (!img_data.empty() &&
                               (img_data.back() == '\r' || img_data.back() == '\n')) {
                            img_data.pop_back();
                        }

                        std::vector<uint8_t> data(img_data.begin(), img_data.end());
                        return cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
                    }
                }
            }
        }
    }

    // 情况 3: JSON body 中包含 base64 编码的图像
    // 尝试解析 JSON 中的 image_base64 字段
    try {
        auto json_body = nlohmann::json::parse(body);
        if (json_body.find("image_base64") != json_body.end()) {
            std::string encoded = json_body["image_base64"];
            std::string decoded = base64_decode(encoded);
            std::vector<uint8_t> data(decoded.begin(), decoded.end());
            return cv::imdecode(cv::Mat(data), cv::IMREAD_COLOR);
        }
    } catch (...) {
        // JSON 解析失败，继续尝试其他格式
    }

    return cv::Mat();
}

HttpResponse RequestHandlers::HandleInfer(const HttpRequest& request) {
    // 提取图像
    cv::Mat image = ExtractImage(request.body, request.content_type);
    if (image.empty()) {
        return HandleError(400, "Failed to decode image");
    }

    LOG_INFO("Received inference request: image={}x{}", image.cols, image.rows);

    // 构建推理请求
    uint64_t request_id = next_request_id_++;
    PendingRequest pending_request(request_id, image, "yolov8");

    // 提交推理（异步）
    auto future = service_->SubmitRequest(std::move(pending_request));

    // 等待结果
    InferenceResponse response = future.get();

    // 序列化为 JSON
    nlohmann::json json_response;
    json_response["request_id"] = response.request_id;
    json_response["status"] = response.status.ok() ? "ok" : "error";

    if (!response.status.ok()) {
        json_response["error"] = response.status.ToString();
    }

    // 添加检测结果
    nlohmann::json detections = nlohmann::json::array();
    for (const auto& det : response.detections) {
        nlohmann::json det_json;
        det_json["class_id"] = det.class_id;
        det_json["class_name"] = det.class_name;
        det_json["confidence"] = det.confidence;
        det_json["bbox"] = {
            {"x", det.x_center},
            {"y", det.y_center},
            {"width", det.width},
            {"height", det.height}
        };
        detections.push_back(det_json);
    }
    json_response["detections"] = detections;
    json_response["num_detections"] = response.detections.size();

    // 添加性能指标
    nlohmann::json timing;
    timing["inference_time_ms"] = response.inference_time_ms;
    timing["preprocessing_time_ms"] = response.preprocessing_time_ms;
    timing["postprocessing_time_ms"] = response.postprocessing_time_ms;
    json_response["timing"] = timing;

    HttpResponse response_http;
    response_http.status_code = 200;
    response_http.body = json_response.dump(2); // 2-space indent

    return response_http;
}

HttpResponse RequestHandlers::HandleHealth() {
    nlohmann::json json;
    json["status"] = "healthy";
    json["service"] = "inference_server";
    json["version"] = "1.0.0";

    HttpResponse response;
    response.body = json.dump(2);
    return response;
}

HttpResponse RequestHandlers::HandleStats() {
    // 获取服务统计
    auto stats = service_->GetBatchStats();

    nlohmann::json json;
    json["total_requests"] = stats.total_requests;
    json["total_batches"] = stats.total_batches;
    json["avg_batch_size"] = stats.avg_batch_size;
    json["pending_requests"] = service_->GetPendingCount();

    HttpResponse response;
    response.body = json.dump(2);
    return response;
}

HttpResponse RequestHandlers::HandleNotFound() {
    return HandleError(404, "Not Found");
}

HttpResponse RequestHandlers::HandleError(int status_code, const std::string& message) {
    nlohmann::json json;
    json["error"] = message;
    json["status_code"] = status_code;

    HttpResponse response;
    response.status_code = status_code;

    // 状态文本
    switch (status_code) {
        case 400: response.status_text = "Bad Request"; break;
        case 404: response.status_text = "Not Found"; break;
        case 500: response.status_text = "Internal Server Error"; break;
        default: response.status_text = "Error";
    }

    response.body = json.dump(2);
    return response;
}

} // namespace inference
