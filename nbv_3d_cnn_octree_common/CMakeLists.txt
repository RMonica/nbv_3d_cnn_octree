cmake_minimum_required(VERSION 3.0.2)
project(nbv_3d_cnn_octree_common)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES nbv_3d_cnn_octree_common
  CATKIN_DEPENDS roscpp
  DEPENDS OpenCV
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(nbv_3d_cnn_octree_common
  src/octree_load_save.cpp
)

target_link_libraries(nbv_3d_cnn_octree_common
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(nbv_3d_cnn_octree_common ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

