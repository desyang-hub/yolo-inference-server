/**
 * @FilePath     : /yolo-onnx-inference/include/inference/server/HttpContext.hpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-06-12 21:46:20
 * @LastEditors  : desyang
 * @LastEditTime : 2026-06-12 21:47:21
**/
#pragma once

#include "Handlers.hpp"

namespace inference
{


class HttpContext {
public:
    enum class HttpRequestParseState {
        kExpectRequestLine,
        kExpectHeaders,
        kExpectBody,
        kGotAll
    };

    HttpContext() : state_(HttpRequestParseState::kExpectRequestLine) {}

    // 核心解析函数：从 buffer 中提取数据
    bool parseRequest(muduo::net::Buffer* buf, muduo::Timestamp receiveTime);

    bool gotAll() const { return state_ == HttpRequestParseState::kGotAll; }
    void reset() {
        state_ = HttpRequestParseState::kExpectRequestLine;
        // request_.reset(); // 清空请求对象
    }

    const HttpRequest& request() const { return request_; }

private:
    bool processRequestLine(const char* begin, const char* end);

    HttpRequestParseState state_;
    HttpRequest request_;
};


} // namespace inference