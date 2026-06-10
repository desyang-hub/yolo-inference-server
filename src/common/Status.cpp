/**
 * @file Status.cpp
 * @brief 状态码实现
 */

#include "inference/common/Status.hpp"

namespace inference {

std::string StatusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::Ok: return "OK";
        case StatusCode::InvalidArgs: return "Invalid Argument";
        case StatusCode::NotFound: return "Not Found";
        case StatusCode::AlreadyExists: return "Already Exists";
        case StatusCode::Unavailable: return "Unavailable";
        case StatusCode::DataLoss: return "Data Loss";
        case StatusCode::DeadlineExceeded: return "Deadline Exceeded";
        case StatusCode::ResourceExhausted: return "Resource Exhausted";
        case StatusCode::Internal: return "Internal Error";
        default: return "Unknown Error";
    }
}

std::string Status::ToString() const {
    std::string result = StatusCodeToString(code_);
    if (!message_.empty()) {
        result += ": ";
        result += message_;
    }
    return result;
}

} // namespace inference
