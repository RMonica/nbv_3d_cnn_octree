#!/usr/bin/python3

import nbv_3d_cnn_msgs.msg as nbv_3d_cnn_msgs
import std_msgs.msg as std_msgs
from rospy.numpy_msg import numpy_msg

import os
import sys
import datetime
import time
from enum import Enum

import numpy as np
import math
import pickle
import struct
import subprocess

import rospy
import cv_bridge
import cv2

class Memtest:
  def __init__(self):
    self.model = None

    self.load_checkpoint_file = rospy.get_param('~load_checkpoint_file', '')

    self.first_test_image = rospy.get_param('~first_test_image', 120)
    self.last_test_image = rospy.get_param('~last_test_image', 180)

    self.input_prefix = rospy.get_param('~input_prefix', '')

    self.is_3d = rospy.get_param('~is_3d', False)
    self.dims = 3 if self.is_3d else 2

    self.crop_image = rospy.get_param('~crop_image', '')

    self.model_type = rospy.get_param('~model_type', "sparse_resnet")

    self.base_channels = rospy.get_param('~base_channels', 16)
    self.resblock_num = rospy.get_param('~resblock_num', 3)
    self.label_hidden_channels = rospy.get_param("~label_hidden_channels", 16)
    self.max_levels = rospy.get_param('~max_levels', 3)
    self.max_channels = rospy.get_param('~max_channels', 1000000000)

    self.engine = rospy.get_param('~engine', "pytorch")

    self.dry_run = rospy.get_param('~dry_run', False)

    self.max_memory_mb = rospy.get_param('~max_memory_mb', 1024)
    self.min_memory_mb = rospy.get_param('~min_memory_mb', 64)
    self.memory_precision_mb = rospy.get_param('~memory_precision_mb', 32)

    pass

  def test_all(self):
    upper_bound = self.max_memory_mb
    lower_bound = self.min_memory_mb
    while upper_bound - lower_bound >= self.memory_precision_mb:
      mean = (upper_bound + lower_bound) / 2.0
      r = self.sparse_octree_memtest(mean)
      if r:
        upper_bound = mean
      else:
        lower_bound = mean
      if rospy.is_shutdown():
        return
    rospy.loginfo('sparse_octree_memtest: upper bound is %f' % upper_bound)

  def sparse_octree_memtest(self, memory_mb):
    rospy.loginfo('sparse_octree_memtest: start.')

    parameters = ["rosrun", "nbv_3d_cnn_octree", "octree_sparse_pytorch_memtest_kernel.py"]

    parameters.append("_max_memory_mb:=%f"         % memory_mb)
    parameters.append("_first_test_image:=%d"      % self.first_test_image)
    parameters.append("_last_test_image:=%d"       % self.last_test_image)
    parameters.append("_load_checkpoint_file:=%s"  % self.load_checkpoint_file)
    parameters.append("_input_prefix:=%s"          % self.input_prefix)
    parameters.append("_base_channels:=%s"         % self.base_channels)
    parameters.append("_max_channels:=%s"          % self.max_channels)
    parameters.append("_resblock_num:=%s"          % self.resblock_num)
    parameters.append("_label_hidden_channels:=%s" % self.label_hidden_channels)
    parameters.append("_max_levels:=%s"            % self.max_levels)
    parameters.append("_model_type:=%s"            % self.model_type)
    parameters.append("_is_3d:=%s"                 % self.is_3d)
    parameters.append("_crop_image:=%s"            % self.crop_image)
    parameters.append("_engine:=%s"                % self.engine)
    parameters.append("_dry_run:=%s"                % ("true" if self.dry_run else "false"))

    rospy.loginfo("sparse_octree_memtest: max_memory_mb was set to %d" % memory_mb)
    rospy.loginfo("sparse_octree_memtest: launching command " + str(parameters))

    result = subprocess.run(parameters)

    rospy.loginfo("sparse_octree_memtest: return code is %d" % result.returncode)
    return result.returncode == 0

  pass # class

rospy.init_node('octree_sparse_pytorch_memtest')

memtest = Memtest()
memtest.test_all()
