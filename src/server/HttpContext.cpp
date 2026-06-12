/**
 * @FilePath     : /yolo-onnx-inference/src/server/HttpContext.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-06-12 21:59:30
 * @LastEditors  : desyang
 * @LastEditTime : 2026-06-12 22:06:06
**/
#include "inference/server/HttpContext.hpp"
#include <algorithm>
#include <iostream> // 仅用于调试，实际可替换为日志宏

namespace inference
{

// 解析请求行：GET /path HTTP/1.1
bool HttpContext::processRequestLine(const char* begin, const char* end) {
    bool succeed = false;
    const char* start = begin;
    
    // 1. 解析请求方法 (Method)
    const char* space = std::find(start, end, ' ');
    if (space != end && request_.setMethod(start, space)) {
        start = space + 1;
        
        // 2. 解析请求路径 (Path) 和查询参数 (Query)
        space = std::find(start, end, ' ');
        if (space != end) {
            const char* question = std::find(start, space, '?');
            if (question != space) {
                request_.setPath(start, question);
                request_.setQuery(question, space);
            } else {
                request_.setPath(start, space);
            }
            start = space + 1;
            
            // 3. 解析协议版本 (Version)
            succeed = end - start == 8 && std::equal(start, end - 1, "HTTP/1.");
            if (succeed) {
                if (*(end - 1) == '1') {
                    request_.setVersion(HttpRequest::kHttp11);
                } else if (*(end - 1) == '0') {
                    request_.setVersion(HttpRequest::kHttp10);
                } else {
                    succeed = false;
                }
            }
        }
    }
    return succeed;
}

// 核心解析函数：从 Buffer 中提取数据并驱动状态机
bool HttpContext::parseRequest(muduo::net::Buffer* buf, muduo::Timestamp receiveTime) {
    bool ok = true;
    bool hasMore = true;
    
    while (hasMore) {
        // 阶段1：解析请求行
        if (state_ == HttpContext::HttpRequestParseState::kExpectRequestLine) {
            const char* crlf = buf->findCRLF();
            if (crlf) {
                ok = processRequestLine(buf->peek(), crlf);
                if (ok) {
                    request_.setReceiveTime(receiveTime);
                    // 消费掉已解析的请求行（包括 \r\n）
                    buf->retrieveUntil(crlf + 2);
                    state_ = HttpContext::HttpRequestParseState::kExpectHeaders;
                } else {
                    hasMore = false; // 格式错误
                }
            } else {
                hasMore = false; // 还没收到完整的请求行，等待下次数据
            }
        } 
        // 阶段2：解析请求头
        else if (state_ == HttpContext::HttpRequestParseState::kExpectHeaders) {
            const char* crlf = buf->findCRLF();
            if (crlf) {
                const char* colon = std::find(buf->peek(), crlf, ':');
                if (colon != crlf) {
                    // 正常的 Header 行，包含冒号
                    request_.addHeader(buf->peek(), colon, crlf);
                } else {
                    // 遇到空行（\r\n），说明 Header 解析完毕
                    if (!request_.getHeader("Content-Length").empty()) {
                        request_.setContentLength(std::stoi(request_.getHeader("Content-Length")));
                    }
                    
                    // 判断是否有请求体 (Body)
                    if (request_.contentLength() == 0) {
                        state_ = HttpContext::HttpRequestParseState::kGotAll; // GET 请求或无 Body 的 POST
                        hasMore = false;
                    } else {
                        state_ = HttpContext::HttpRequestParseState::kExpectBody; // 有 Body，进入下一阶段
                    }
                }
                // 消费掉当前 Header 行
                buf->retrieveUntil(crlf + 2);
            } else {
                hasMore = false; // Header 还没接收完整
            }
        } 
        // 阶段3：解析请求体
        else if (state_ == HttpContext::HttpRequestParseState::kExpectBody) {
            size_t need = request_.contentLength();
            if (buf->readableBytes() >= need) {
                // Body 数据已经全部到达
                request_.setBody(std::string(buf->peek(), need));
                buf->retrieve(need);
                state_ = HttpContext::HttpRequestParseState::kGotAll;
                hasMore = false;
            } else {
                hasMore = false; // Body 数据不够，等待下次数据到来
            }
        }
    }
    return ok;
}


} // namespace inference