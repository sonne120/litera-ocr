cmake_minimum_required(VERSION 3.20)
project(litera_ocr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LITERA_OCR_BUILD_TESTS "Build unit tests" ON)
option(LITERA_OCR_BUILD_CLI "Build command-line tool" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to extracted ONNX Runtime")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
    NO_DEFAULT_PATH REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_library(litera_ocr
    src/ctc_decoder.cpp
    src/preprocess.cpp
    src/db_detector.cpp
    src/crnn_recognizer.cpp
    src/ocr_engine.cpp)

target_include_directories(litera_ocr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(litera_ocr
    PUBLIC opencv_core
    PRIVATE opencv_imgproc opencv_imgcodecs onnxruntime)

target_compile_options(litera_ocr PRIVATE -Wall -Wextra)

if(LITERA_OCR_BUILD_CLI)
    add_executable(litera-ocr cli/main.cpp)
    target_link_libraries(litera-ocr PRIVATE litera_ocr opencv_imgcodecs)
endif()

if(LITERA_OCR_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(litera_ocr_tests
        tests/ctc_decoder_test.cpp
        tests/preprocess_test.cpp
        tests/pipeline_test.cpp)
    target_link_libraries(litera_ocr_tests PRIVATE litera_ocr opencv_imgcodecs
        opencv_imgproc GTest::gtest_main)
    target_compile_definitions(litera_ocr_tests PRIVATE
        LITERA_OCR_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    include(GoogleTest)
    gtest_discover_tests(litera_ocr_tests DISCOVERY_MODE PRE_TEST)
endif()

include(GNUInstallDirs)
install(TARGETS litera_ocr
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
