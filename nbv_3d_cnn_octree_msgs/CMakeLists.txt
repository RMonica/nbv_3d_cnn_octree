cmake_minimum_required(VERSION 3.0.2)
project(nbv_3d_cnn_octree_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  message_generation
  std_msgs
  sensor_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  Octree.msg
  OctreeLevel.msg
)

add_action_files(
  FILES
  PredictOctree.action
  PredictImage.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  actionlib_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES nbv_3d_cnn_msgs
  CATKIN_DEPENDS actionlib_msgs message_runtime std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

#############
## Install ##
#############

#############
## Testing ##
#############

