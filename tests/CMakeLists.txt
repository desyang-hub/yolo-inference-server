cmake_minimum_required(VERSION 3.18)

# 获取 GTest（如果未安装）
if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# 测试可执行文件（不依赖 ONNX Runtime）
add_executable(test_preprocessing
    ../src/preprocessing/ImagePreprocessor.cpp
    ../src/preprocessing/NMS.cpp
    ../src/common/Logger.cpp
    ../src/common/Status.cpp
    test_preprocessing.cpp
)

target_include_directories(test_preprocessing PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
)

target_link_libraries(test_preprocessing PRIVATE
    ${OpenCV_LIBS}
    ${SPDLOG_LIBRARY}
    GTest::gtest_main
    Threads::Threads
)

gtest_discover_tests(test_preprocessing)

# 批调度器测试（仅当 ONNX Runtime 可用时）
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    add_executable(test_batch_scheduler
        ../src/batch/BatchScheduler.cpp
        ../src/model/ModelSession.cpp
        ../src/preprocessing/ImagePreprocessor.cpp
        ../src/preprocessing/NMS.cpp
        ../src/common/Logger.cpp
        ../src/common/Status.cpp
        test_batch_scheduler.cpp
    )

    target_include_directories(test_batch_scheduler PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${NLOHMANN_JSON_INCLUDE_DIR}
        ${SPDLOG_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
    )

    target_compile_definitions(test_batch_scheduler PRIVATE ONNXRUNTIME_FOUND)

    target_link_libraries(test_batch_scheduler PRIVATE
        ${OpenCV_LIBS}
        ${SPDLOG_LIBRARY}
        ${ONNXRUNTIME_LIBRARY}
        GTest::gtest_main
        Threads::Threads
    )

    gtest_discover_tests(test_batch_scheduler)
else()
    message(STATUS "  ONNX Runtime not found. Skipping test_batch_scheduler.")
endif()
