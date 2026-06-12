cmake_minimum_required(VERSION 3.18)
project(inference_server VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# 项目配置
# ============================================================
set(PROJECT_NAME inference_server)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================
# 依赖查找
# ============================================================

# 1. OpenCV (图像处理)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "  OpenCV ${OpenCV_VERSION} found")

# 2. muduo (网络库 - Reactor 模式)
# muduo 安装在系统默认路径，不需要 find_package
set(MUDUO_LIBRARIES muduo_net muduo_base pthread)
message(STATUS "  muduo found (system libraries)")

# 3. ONNX Runtime (推理引擎)
# 初始化为空，避免 CMake 错误
set(ONNX_RUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

# 方式 A: 使用系统安装的 ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h onnxruntime/core/session/onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime onnxruntime_providers_shared)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "  ONNX Runtime found (system: ${ONNXRUNTIME_LIBRARY})")
else()
    # 方式 B: 使用自定义路径（编译时设置）
    if(DEFINED ONNX_RUNTIME_ROOT)
        find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            PATHS ${ONNX_RUNTIME_ROOT}
            PATH_SUFFIXES include include/onnxruntime
            NO_DEFAULT_PATH)
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime libonnxruntime.so
            PATHS ${ONNX_RUNTIME_ROOT}
            PATH_SUFFIXES lib
            NO_DEFAULT_PATH)
        message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

        if (ONNXRUNTIME_INCLUDE_DIR)
            
        endif()

    else()
        if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
            message(STATUS "  ONNX Runtime found (custom: ${ONNXRUNTIME_LIBRARY})")
        else()
            message(WARNING "ONNX Runtime NOT found. Set ONNX_RUNTIME_ROOT or install it.")
            message(WARNING "See docs/install_dependencies.sh for installation instructions.")
        endif()
    endif()
endif()

# 4. nlohmann/json (JSON 解析 - 头文件库)
# 优先使用系统安装的，否则通过 FetchContent 下载
find_path(NLOHMANN_JSON_INCLUDE_DIR NAMES nlohmann/json.hpp)

if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "  Fetching nlohmann/json via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
    set(NLOHMANN_JSON_INCLUDE_DIR ${nlohmann_json_SOURCE_DIR}/include)
endif()
message(STATUS "  nlohmann/json found (${NLOHMANN_JSON_INCLUDE_DIR})")

# 5. spdlog (日志库)
find_path(SPDLOG_INCLUDE_DIR NAMES spdlog/spdlog.h spdlog/fwd/logger.h)
find_library(SPDLOG_LIBRARY NAMES spdlog)

if(SPDLOG_INCLUDE_DIR AND SPDLOG_LIBRARY)
    message(STATUS "  spdlog found (system)")
else()
    message(STATUS "  Fetching spdlog via FetchContent...")
    if(NOT DEFINED FetchContent)
        include(FetchContent)
    endif()
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.14.1
    )
    FetchContent_MakeAvailable(spdlog)
    set(SPDLOG_INCLUDE_DIR ${spdlog_SOURCE_DIR}/include)
    set(SPDLOG_LIBRARY spdlog)
endif()

# 6. Threading
find_package(Threads REQUIRED)
message(STATUS "  Threading support enabled")

# 7. httplib
set(HTTPLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib/)

# ============================================================
# 源码收集
# ============================================================
set(SOURCES
    src/common/Logger.cpp
    src/common/Status.cpp
    src/model/ModelSession.cpp
    src/model/ModelManager.cpp
    src/inference/InferenceService.cpp
    src/batch/BatchScheduler.cpp
    src/server/InferenceServer.cpp
    src/server/Handlers.cpp
    src/preprocessing/ImagePreprocessor.cpp
    src/preprocessing/NMS.cpp
    src/main.cpp
)

set(HEADERS
    include/inference/common/Status.hpp
    include/inference/common/Result.hpp
    include/inference/common/Logger.hpp
    include/inference/model/ModelConfig.hpp
    include/inference/model/ModelSession.hpp
    include/inference/model/ModelManager.hpp
    include/inference/inference/InferenceRequest.hpp
    include/inference/inference/InferenceResponse.hpp
    include/inference/inference/InferenceService.hpp
    include/inference/batch/BatchConfig.hpp
    include/inference/batch/PendingRequest.hpp
    include/inference/batch/BatchScheduler.hpp
    include/inference/server/ServerConfig.hpp
    include/inference/server/Handlers.hpp
    include/inference/server/InferenceServer.hpp
    include/inference/thread/TaskQueue.hpp
    include/inference/thread/ThreadUtils.hpp
    include/inference/preprocessing/ImagePreprocessor.hpp
    include/inference/preprocessing/NMS.hpp
    include/inference/memory/ObjectPool.hpp
    include/inference/memory/MemoryPool.hpp
)

# ============================================================
# 可执行文件
# ============================================================
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
    ${HTTPLIB_INCLUDE_DIR}
)

# ONNX Runtime 包含目录（如果找到）
if(ONNXRUNTIME_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNXRUNTIME_FOUND)
endif()

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${MUDUO_LIBRARIES}
    ${SPDLOG_LIBRARY}
    Threads::Threads
    dl
)

# ONNX Runtime 链接（如果找到）
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNXRUNTIME_FOUND)
endif()

# nlohmann/json 是头文件库，无需链接
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)
endif()

# target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer -O1)
# target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)

# target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
# target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=thread)

# ============================================================
# 安装规则
# ============================================================
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# ============================================================
# 测试
# ============================================================
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================
# 框架测试目标（不依赖 ONNX Runtime）
# ============================================================
# 当 ONNX Runtime 未安装时，可以编译和测试预处理/NMS 等组件
add_executable(test_framework
    src/common/Logger.cpp
    src/common/Status.cpp
    src/preprocessing/ImagePreprocessor.cpp
    src/preprocessing/NMS.cpp
    tests/test_preprocessing.cpp
)

target_include_directories(test_framework PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
)

target_link_libraries(test_framework PRIVATE
    ${OpenCV_LIBS}
    ${SPDLOG_LIBRARY}
    GTest::gtest_main
    Threads::Threads
)

# 如果 GTest 未安装，跳过
if(NOT TARGET GTest::gtest_main)
    message(STATUS "  GTest not found. Skipping test_framework target.")
    # 使用 FetchContent 获取 GTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    target_link_libraries(test_framework PRIVATE GTest::gtest_main)
endif()

include(GoogleTest)
gtest_discover_tests(test_framework)

# ============================================================
# 配置总结
# ============================================================
message(STATUS "")
message(STATUS "=== ${PROJECT_NAME} Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "===================================")
message(STATUS "")
