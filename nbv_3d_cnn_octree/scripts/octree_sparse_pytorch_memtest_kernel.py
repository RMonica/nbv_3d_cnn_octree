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

import rospy
import cv_bridge
import cv2

import torch

import model_factory
import octree_save_load

import enum_config

class Memtest:

  def __init__(self):
    self.model = None

    self.load_checkpoint_file = rospy.get_param('~load_checkpoint_file', '')

    self.first_test_image = rospy.get_param('~first_test_image', 120)
    self.last_test_image = rospy.get_param('~last_test_image', 180)

    self.input_prefix = rospy.get_param('~input_prefix', '')

    self.is_3d = rospy.get_param('~is_3d', False)
    self.dims = 3 if self.is_3d else 2

    self.crop_image = str(rospy.get_param('~crop_image', "1250 1250"))
    self.crop_image = list(filter(lambda s: (s != ""), self.crop_image.split(" ")))
    self.crop_image = [int(x) for x in self.crop_image]
    if len(self.crop_image) != 2 and not self.is_3d:
      rospy.logfatal("Expected two values in crop_image parameter, got " + str(self.crop_image))
      exit(1)
    if len(self.crop_image) != 3 and self.is_3d:
      rospy.logfatal("Expected three values in crop_image parameter, got " + str(self.crop_image))
      exit(1)
    rospy.loginfo("image will be cropped at %s" % str(self.crop_image))

    self.model_type = rospy.get_param('~model_type', enum_config.ModelType.SPARSE_RESNET)
    self.model_type = enum_config.ModelType(self.model_type)

    if self.model_type == enum_config.ModelType.SPARSE_RESNET:
      self.input_type = enum_config.InputType.OCTREE
    elif self.model_type == enum_config.ModelType.SPARSE_ENC_DEC:
      self.input_type = enum_config.InputType.OCTREE
    elif self.model_type == enum_config.ModelType.RESNET:
      self.input_type = enum_config.InputType.IMAGE
    elif self.model_type == enum_config.ModelType.ENC_DEC:
      self.input_type = enum_config.InputType.IMAGE
    else:
      rospy.logfatal("Unknown model_type parameter: " + self.model_type)
      exit(1)

    self.engine = rospy.get_param('~engine', "pytorch")
    self.engine = enum_config.EngineType(self.engine)

    self.dry_run = rospy.get_param('~dry_run', False)

    self.base_channels = rospy.get_param('~base_channels', 16)
    self.resblock_num = rospy.get_param('~resblock_num', 3)
    self.label_hidden_channels = rospy.get_param("~label_hidden_channels", 16)
    self.octree_depth = rospy.get_param('~max_levels', 3)
    self.max_channels = rospy.get_param('~max_channels', 1000000000)
    pass

  def sparse_octree_memtest(self):
    rospy.loginfo('sparse_octree_memtest: start.')

    #resblock_num = 3
    resblock_num = self.resblock_num
    rospy.loginfo("sparse_octree_test: Engine is %s" % str(self.engine))
    model = model_factory.get_model(model_type=self.model_type, engine=self.engine,
                                    nin=2, nout=1, octree_depth=self.octree_depth, resblock_num=self.resblock_num, max_channels=self.max_channels,
                                    base_channels=self.base_channels, label_hidden_channels=self.label_hidden_channels,
                                    is_3d=self.is_3d)

    if model is None:
      rospy.logfatal("Model not implemented with the given configuration.")
      exit(1)

    rospy.loginfo("Model parameters are: ")
    for name, p in model.named_parameters():
      rospy.loginfo("  " + name)

    if self.load_checkpoint_file != "":
      model.load_state_dict(torch.load(self.load_checkpoint_file))

    model.eval()

    avg_mem       = 0
    avg_pred_time = 0
    avg_loss      = 0
    avg_counter   = 0
    avg_total_output_values = 0

    with torch.no_grad():
      for image_num in range(self.first_test_image, self.last_test_image):
        torch.cuda.empty_cache()
        rospy.loginfo("TESTING IMAGE %d" % image_num)
        gt_octree_filename = self.input_prefix + str(image_num) + "_gt_octree.octree";
        input_octree_filename = self.input_prefix + str(image_num) + "_input_octree.octree";
        with open(gt_octree_filename, "rb") as ifile:
          gt_outputs, gt_masks, _, _, _, _ = octree_save_load.load_octree_from_file(ifile)
        with open(input_octree_filename, "rb") as ifile:
          inputs, masks, uninteresting_masks, _, _, _ = octree_save_load.load_octree_from_file(ifile)

        if self.input_type == enum_config.InputType.IMAGE:
          gt_output = octree_save_load.octree_to_pytorch_image(gt_outputs, gt_masks, self.crop_image).unsqueeze(0)
          input = octree_save_load.octree_to_pytorch_image(inputs, masks, self.crop_image).unsqueeze(0)
          gt_mask = octree_save_load.octree_to_pytorch_image(gt_masks, gt_masks, self.crop_image).unsqueeze(0)

        time_before_prediction = time.perf_counter()
        if self.dry_run:
          total_output_values = 0.0
        else:
          if self.input_type == enum_config.InputType.OCTREE:
            outputs, output_masks, output_logits, _ = model(inputs, masks, forced_masks=None, uninteresting_masks=uninteresting_masks)
            total_output_values = np.sum([om.values().shape[0] for om in output_masks])
          if self.input_type == enum_config.InputType.IMAGE:
            output = model(input)
            total_output_values = 0.0
        time_after_prediction = time.perf_counter()

        pred_time = time_after_prediction - time_before_prediction
        rospy.loginfo("prediction: %fs" % pred_time)
        display_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        rospy.loginfo("model memory MB: " + str(display_mem))

        input = None
        inputs = None
        masks = None
        output = None
        outputs = None
        output_masks = None
        output_logits = None

        if rospy.is_shutdown():
          return
        pass
      pass
    pass
  pass # class

rospy.init_node('octree_sparse_pytorch_memtest_kernel', anonymous=True)

if torch.cuda.is_available():
  rospy.loginfo("CUDA is available")
else:
  rospy.loginfo("CUDA is NOT available")
  exit(1)

torch.manual_seed(10)

#torch.autograd.set_detect_anomaly(True)

rospy.loginfo("CUDA total memory: %d MB" % (torch.cuda.get_device_properties(0).total_memory / (1024*1024)))

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
max_memory_fraction = min(float(max_memory_mb) * 1024*1024 / float(torch.cuda.get_device_properties(0).total_memory), 1.0)
rospy.loginfo("CUDA max memory fraction %f (%d MB)" % (max_memory_fraction, max_memory_mb))
torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
torch.set_default_device('cuda')

trainer = Memtest()
trainer.sparse_octree_memtest();
