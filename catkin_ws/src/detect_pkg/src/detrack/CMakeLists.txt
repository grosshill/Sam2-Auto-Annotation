cmake_minimum_required(VERSION 3.5)
project(detrack_node)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++17 )
ADD_COMPILE_OPTIONS(-std=c++17 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    visualization_msgs
    geometry_msgs
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
    include
    SYSTEM
    third_party
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -O3 -Wall") # -Wextra -Werror

add_subdirectory(bytetrack)
add_subdirectory(YOLO26_TensorRT_ONNX)



add_library( kalman_filter
    STATIC
    src/ekf.cpp
)

add_executable(ekf
     ekf_node.cpp
)

target_link_libraries(ekf
        kalman_filter
    ${catkin_LIBRARIES}
)

add_executable(detrack_node
     detrack_node.cpp
)

target_include_directories(detrack_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack/include
    ${CMAKE_CURRENT_SOURCE_DIR}/YOLO26_TensorRT_ONNX/src
)

target_link_libraries(detrack_node
    PRIVATE
        bytetrack_core
        YoloDet26
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
)