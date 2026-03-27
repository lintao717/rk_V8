# 指定CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
PROJECT(rk_cmake_project)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定交叉编译器（根据实际情况修改路径）
set(CMAKE_C_COMPILER /home/lt/tspi_SDK/buildroot/output/rockchip_rk3566/host/bin/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /home/lt/tspi_SDK/buildroot/output/rockchip_rk3566/host/bin/aarch64-buildroot-linux-gnu-g++)

# 添加头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    # mpi 相关头文件
    ${CMAKE_SOURCE_DIR}/include/mpi/example
    ${CMAKE_SOURCE_DIR}/include/mpi/sdk
    # rknn 相关头文件
    ${CMAKE_SOURCE_DIR}/include/rknn/rknpu2
    ${CMAKE_SOURCE_DIR}/include/rknn/utils
    ${CMAKE_SOURCE_DIR}/include/rknn/yolov8
    ${CMAKE_SOURCE_DIR}/include/rknn/librga
    ${CMAKE_SOURCE_DIR}/include/rknn/stb_image
    ${CMAKE_SOURCE_DIR}/include/rknn/jpeg_turbo
    # ZLMediaKit 相关头文件
    ${CMAKE_SOURCE_DIR}/include/ZLMediaKit
)

# 添加源文件
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/rk_mpi.cpp
    ${CMAKE_SOURCE_DIR}/src/rknn/utils/file_utils.c
    ${CMAKE_SOURCE_DIR}/src/rknn/utils/image_drawing.c
    ${CMAKE_SOURCE_DIR}/src/rknn/utils/image_utils.c
    ${CMAKE_SOURCE_DIR}/src/rknn/yolov8/postprocess.cc
    ${CMAKE_SOURCE_DIR}/src/rknn/yolov8/yolov8.cc
)

# 添加库文件路径
link_directories(
    ${CMAKE_SOURCE_DIR}/lib
)

# 添加目标可执行文件
add_executable(rk_cmake_project ${SOURCES})

# 添加链接库
target_link_libraries(rk_cmake_project 
    pthread
    rockit
    rt_test_comm
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    rknnrt
    rga
    turbojpeg
    mk_api
)
