cmake_minimum_required(VERSION 3.12)
project(motion_segmentation LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")


find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(PkgConfig REQUIRED)


include_directories(include
                    thirdparty/gco
                    thirdparty/delaunator/include
                    ${OpenCV_INCLUDE_DIRS})


set(HEADERS
    include/motionseg/MotionSeg.h
    thirdparty/gco/energy.h
    thirdparty/gco/block.h
    thirdparty/gco/GCoptimization.h
    thirdparty/gco/graph.h
    thirdparty/gco/LinkedBlockList.h
    thirdparty/delaunator/include/delaunator.hpp)

set(SOURCES
    src/MotionSeg.cpp
    thirdparty/gco/GCoptimization.cpp
    thirdparty/gco/graph.cpp
    thirdparty/gco/LinkedBlockList.cpp
    thirdparty/gco/maxflow.cpp)

add_library(${PROJECT_NAME}_LIB ${SOURCES} ${HEADERS})

# 主程序配置
add_executable(motion_segmentation 
    # src/MotionSeg.cpp
    src/MotionSeg_node.cpp)


# # OpenMP支持（如果代码中有OpenMP指令）
find_package(OpenMP REQUIRED)
target_link_libraries(motion_segmentation PRIVATE 
    ${PROJECT_NAME}_LIB
    OpenMP::OpenMP_CXX
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
)