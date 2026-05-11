cmake_minimum_required(VERSION 3.5)

project(bytetrack)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
# include_directories(/usr/local/include/eigen3)
link_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)
# tensorrt
include_directories(/opt/tiger/demo/TensorRT-7.2.3.4/include)
link_directories(/opt/tiger/demo/TensorRT-7.2.3.4/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

set(BYTETRACK_CORE_SOURCES
	src/BYTETracker.cpp
	src/STrack.cpp
	src/kalmanFilter.cpp
	src/lapjv.cpp
	src/utils.cpp
)

add_library(bytetrack_core STATIC ${BYTETRACK_CORE_SOURCES})
target_include_directories(bytetrack_core PUBLIC
	${PROJECT_SOURCE_DIR}/include
	${EIGEN3_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(bytetrack_core PUBLIC ${OpenCV_LIBS})

add_executable(bytetrack src/bytetrack.cpp)
target_link_libraries(bytetrack bytetrack_core nvinfer cudart ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

