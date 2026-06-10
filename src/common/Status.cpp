/**
 * @file Status.cpp
 * @brief 状态码实现
 */

#include "inference/common/Status.hpp"

namespace inference {

std::string StatusCodeToString(Status::kStatusCode code) {
    switch (code) {
        case Status::kStatusCode::kOk: return "OK";
        case Status::kStatusCode::kInvalidArgs: return "Invalid Argument";
        case Status::kStatusCode::kNotFound: return "Not Found";
        case Status::kStatusCode::kAlreadyExists: return "Already Exists";
        case Status::kStatusCode::kUnavailable: return "Unavailable";
        case Status::kStatusCode::kDataLoss: return "Data Loss";
        case Status::kStatusCode::kDeadlineExceeded: return "Deadline Exceeded";
        case Status::kStatusCode::kResourceExhausted: return "Resource Exhausted";
        case Status::kStatusCode::kInternal: return "Internal Error";
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
