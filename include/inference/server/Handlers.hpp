/**
 * @file Handlers.hpp
 * @brief HTTP 请求处理器
 *
 * 学习要点：
 * ============================================
 * HTTP 协议基础
 * ============================================
 *
 * HTTP Request 格式:
 *   POST /infer HTTP/1.1
 *   Host: localhost:8080
 *   Content-Type: multipart/form-data; boundary=----WebKitFormBoundary
 *   Content-Length: 12345
 *
 *   ------WebKitFormBoundary
 *   Content-Disposition: form-data; name="image"; filename="test.jpg"
 *   Content-Type: image/jpeg
 *
 *   <binary image data>
 *   ------WebKitFormBoundary--
 *
 * HTTP Response 格式:
 *   HTTP/1.1 200 OK
 *   Content-Type: application/json
 *   Content-Length: 256
 *
 *   {"request_id": 1, "detections": [...], "inference_time_ms": 12.5}
 *
 * ============================================
 * muduo 网络编程
 * ============================================
 *
 * muduo 使用 Reactor 模式：
 * - EventLoop: 事件循环，处理 IO 事件
 * - TcpServer: 管理多个 TcpConnection
 * - TcpConnection: 每个连接的读写事件
 * - Buffer: 网络缓冲区
 *
 * 消息处理流程:
 * 1. 连接建立 → connectionCallback
 * 2. 数据到达 → messageCallback → 解析 HTTP 请求
 * 3. 处理请求 → 调用推理服务
 * 4. 发送响应 → 序列化 JSON → 写入 Buffer
 * 5. 连接关闭 → connectionCallback
 */

#pragma once

#include <string>

#include "inference/inference/InferenceService.hpp"

// muduo headers
#include <muduo/net/TcpServer.h>
#include <muduo/net/TcpConnection.h>
#include <muduo/net/Buffer.h>

namespace inference {

/**
 * @brief HTTP 请求结构（简化版）
 */
// struct HttpRequest {
//     std::string method;            // GET/POST
//     std::string path;              // /infer, /health, /stats
//     std::string content_type;      // multipart/form-data, application/json
//     std::string body;              // 请求体（原始数据）
//     cv::Mat image_data;            // 解析后的图像数据
//     bool parsed = false;           // 是否解析成功
// };

class HttpRequest {
public:
    enum Method { kInvalid, kGet, kPost, kHead, kPut, kDelete };
    enum Version { kUnknown, kHttp10, kHttp11 };

    bool setMethod(const char* start, const char* end) {
        std::string m(start, end);
        if (m == "GET") method_ = kGet;
        else if (m == "POST") method_ = kPost;
        else if (m == "HEAD") method_ = kHead;
        else if (m == "PUT") method_ = kPut;
        else if (m == "DELETE") method_ = kDelete;
        else method_ = kInvalid;
        return method_ != kInvalid;
    }

    void setPath(const char* start, const char* end) { path_.assign(start, end); }
    void setQuery(const char* start, const char* end) { query_.assign(start, end); }
    void setVersion(Version v) { version_ = v; }
    void setReceiveTime(muduo::Timestamp t) { receiveTime_ = t; }
    
    // 解析 Header 的辅助函数
    void addHeader(const char* start, const char* colon, const char* end) {
        std::string field(start, colon);
        ++colon;
        // 去除冒号后的空格
        while (colon < end && isspace(*colon)) ++colon;
        std::string value(colon, end);
        // 去除尾部空格
        while (!value.empty() && isspace(value[value.size() - 1])) {
            value.resize(value.size() - 1);
        }
        headers_[field] = value;
    }

    // std::string getHeader(const std::string& field) const {
    //     auto it = headers_.find(field);
    //     return (it != headers_.end()) ? it->second : "";
    // }

    // 在 HttpRequest 类中优化 getHeader
    std::string getHeader(const std::string& field) const {
        // 1. 先尝试精确匹配
        auto it = headers_.find(field);
        if (it != headers_.end()) return it->second;

        // 2. 如果没找到，进行大小写不敏感查找
        for (const auto& header : headers_) {
            if (header.first.size() == field.size() &&
                std::equal(header.first.begin(), header.first.end(), field.begin(), 
                        [](char a, char b) { return std::tolower(a) == std::tolower(b); })) {
                return header.second;
            }
        }
        return "";
    }

    void setContentLength(size_t len) { contentLength_ = len; }
    size_t contentLength() const { return contentLength_; }
    void setBody(const std::string& body) { body_ = body; }
    
    void reset() { /* 清空所有成员变量 */ }

    // Getter 方法
    Method method() const { return method_; }
    const std::string& path() const { return path_; }
    const std::string& body() const { return body_; }

private:
    Method method_ = kInvalid;
    Version version_ = kUnknown;
    std::string path_;
    std::string query_;
    muduo::Timestamp receiveTime_;
    std::map<std::string, std::string> headers_;
    size_t contentLength_ = 0;
    std::string body_;
};



/**
 * @brief HTTP 响应结构
 */
struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::string content_type = "application/json";
    std::string body;

    /** 转换为完整的 HTTP 响应字符串 */
    std::string Serialize() const;
};

/**
 * @brief HTTP 请求处理器
 *
 * 负责：
 * 1. 解析 HTTP 请求
 * 2. 路由到对应的处理函数
 * 3. 序列化响应
 */
class RequestHandlers {
public:
    explicit RequestHandlers(InferenceService* service);

    /**
     * @brief muduo 连接回调
     */
    void OnConnection(const muduo::net::TcpConnectionPtr& conn);

    /**
     * @brief muduo 消息回调
     */
    void OnMessage(const muduo::net::TcpConnectionPtr& conn,
                   muduo::net::Buffer* buffer,
                   muduo::Timestamp receiveTime);

private:
    /**
     * @brief 解析 HTTP 请求
     */
    HttpRequest ParseRequest(const std::string& data);

    /**
     * @brief 提取图像数据（multipart/form-data）
     */
    cv::Mat ExtractImage(const std::string& body, const std::string& content_type);

    /**
     * @brief 处理推理请求
     */
    HttpResponse HandleInfer(const HttpRequest& request);

    /**
     * @brief 处理健康检查
     */
    HttpResponse HandleHealth();

    /**
     * @brief 处理统计信息
     */
    HttpResponse HandleStats();

    /**
     * @brief 处理未知路径
     */
    HttpResponse HandleNotFound();

    /**
     * @brief 处理错误
     */
    HttpResponse HandleError(int status_code, const std::string& message);

    InferenceService* service_ = nullptr;
    uint64_t next_request_id_ = 1;
};

} // namespace inference
