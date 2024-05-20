cmake_minimum_required(VERSION 3.0.2)
project(nbv_3d_cnn_octree)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  eigen_conversions
  nbv_3d_cnn_msgs
  nbv_3d_cnn_octree_msgs
  pcl_conversions
  roscpp
  rospy
  sensor_msgs
  std_msgs
  rmonica_voxelgrid_common
  visualization_msgs
  nbv_3d_cnn_octree_common
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)
find_package(PCL REQUIRED)

find_package(OpenCL REQUIRED)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=110)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)

################################################
## Declare ROS messages, services and actions ##
################################################

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES nbv_3d_cnn_octree
#  CATKIN_DEPENDS actionlib actionlib_msgs cv_bridge eigen_conversions nbv_3d_cnn_msgs pcl_conversions roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/o-cnn/octree/octree"
)
link_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/o-cnn/octree/build"
)

add_executable(octree_to_points
  src/octree_to_points.cpp
)
target_link_libraries(octree_to_points
  ${catkin_LIBRARIES}
  octree_lib
)

add_executable(image_to_octree
  src/image_to_octree.cpp
)

target_link_libraries(image_to_octree
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(octree_to_image
  src/octree_to_image.cpp
)

target_link_libraries(octree_to_image
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(octree_to_visualization_msgs
  src/octree_to_visualization_msgs.cpp
)

target_link_libraries(octree_to_visualization_msgs
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(voxelgrid_predict
  src/voxelgrid_predict.cpp
)

target_link_libraries(voxelgrid_predict
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

#############
## Testing ##
#############

