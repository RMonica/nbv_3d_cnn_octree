#!/usr/bin/python3

import std_msgs.msg as std_msgs
from rospy.numpy_msg import numpy_msg
import nbv_3d_cnn_octree_msgs.msg as nbv_3d_cnn_octree_msgs

import enum_config

import numpy as np
import math
import time

import rospy
import actionlib
import cv_bridge

import octree_save_load
import model_factory

import torch

class PredictAction(object):
  def __init__(self):

    self.last_input_shape = [0, 0]
    self.model = None

    self.checkpoint_file = rospy.get_param('~checkpoint_file', '')
    self.action_name = rospy.get_param('~action_name', '~predict')
    self.model_type = rospy.get_param('~model_type', '')
    self.model_type = enum_config.ModelType(self.model_type)
    self.is_3d = rospy.get_param('~is_3d', False)

    self.dims = 3 if self.is_3d else 2

    self.base_channels = rospy.get_param('~base_channels', 4)
    self.max_levels = rospy.get_param('~max_levels', 3)
    self.label_hidden_channels = rospy.get_param('~label_hidden_channels', 16)
    self.resblock_num = rospy.get_param('~resblock_num', 3)
    self.max_channels = rospy.get_param('~max_channels', 1000000000)

    self.engine = rospy.get_param('~engine', enum_config.EngineType.TORCHSPARSE)
    self.engine = enum_config.EngineType(self.engine)

    if self.model_type == enum_config.ModelType.SPARSE_RESNET:
      self.input_type = enum_config.InputType.OCTREE
    elif self.model_type == enum_config.ModelType.SPARSE_ENC_DEC:
      self.input_type = enum_config.InputType.OCTREE
    elif self.model_type == enum_config.ModelType.RESNET:
      self.input_type = enum_config.InputType.IMAGE
    elif self.model_type == enum_config.ModelType.ENC_DEC:
      self.input_type = enum_config.InputType.IMAGE
    else:
      rospy.logfatal("nbv_3d_cnn_octree_predict: Unknown model_type parameter: " + self.model_type)
      exit(1)

    if self.input_type == enum_config.InputType.IMAGE:
      self.action_server = actionlib.SimpleActionServer(self.action_name, nbv_3d_cnn_octree_msgs.PredictImageAction,
                                                        execute_cb=self.on_predict_image, auto_start=False)
    if self.input_type == enum_config.InputType.OCTREE:
      self.action_server = actionlib.SimpleActionServer(self.action_name, nbv_3d_cnn_octree_msgs.PredictOctreeAction,
                                                        execute_cb=self.on_predict_octree, auto_start=False)
    self.action_server.start()
    rospy.loginfo('nbv_3d_cnn_octree_predict: action \'%s\' started.' % self.action_name)
    pass

  def init_model(self):
    rospy.loginfo("octree_predict: init model.")

    octree_depth = self.max_levels

    resblock_num = self.resblock_num
    model = None
    rospy.loginfo("octree_predict: Engine is %s" % str(self.engine))
    model = model_factory.get_model(model_type=self.model_type, engine=self.engine,
                                    nin=2, nout=1, octree_depth=octree_depth, resblock_num=self.resblock_num, max_channels=self.max_channels,
                                    base_channels=self.base_channels, label_hidden_channels=self.label_hidden_channels,
                                    is_3d=self.is_3d)

    if model is None:
      rospy.logfatal("octree_predict: Model not implemented with the given configuration.")
      exit(1)

    if self.checkpoint_file == "":
      rospy.logerr("octree_predict: Checkpoint file not set, using uninitialized model.")

    rospy.loginfo("octree_predict: Loading model from file %s" % str(self.checkpoint_file))
    errors = model.load_state_dict(torch.load(self.checkpoint_file))
    if len(errors.missing_keys) != 0 or len(errors.unexpected_keys) != 0:
      rospy.logfatal("octree_predict: errors while loading model: " + str(errors))
      exit(1)
    rospy.loginfo("octree_predict: Model loaded.")
    return model

  def octree_from_message(self, message):
    is_3d = message.is_3d
    if self.is_3d != is_3d:
      rospy.logerr("nbv_3d_cnn_octree_predict: octree_from_message: initialized with is_3d=%d, received message with is_3d=%d." %
                   (int(self.is_3d), int(is_3d)))
      return None, None
    dims = 3 if is_3d else 2
    num_channels = int(message.num_channels)
    octree_depth = len(message.levels)

    imgs = []
    masks = []

    for l in range(0, octree_depth):
      level = message.levels[l]
      size = level.size
      if (len(size) != dims):
        rospy.logerr("nbv_3d_cnn_octree_predict: octree_from_message: octree dims is %d, but size contains only %d elements." %
                     (dims, len(size)))
        return None, None
      img_size = list(size) + [num_channels, ]
      mask_size = list(size) + [1, ]
      values = level.values
      values = np.asarray(values, dtype=np.float32)
      values = values.reshape([-1, num_channels])
      ones_shape = list(values.shape)
      ones_shape[-1] = 1 # replace last dimension with 1
      ones = np.ones(ones_shape, dtype=np.float32)
      indices = level.indices
      indices = np.asarray(indices, dtype=np.int64)
      indices = indices.reshape([-1, dims])
      indices = indices.transpose(1, 0)
      # add the batch dimension (all zeros)
      batch_index = np.expand_dims(np.zeros(indices[0].shape, dtype=np.int64), axis=0)
      img_size = [1, ] + img_size
      mask_size = [1, ] + mask_size
      indices = np.concatenate([batch_index, indices], axis=0)

      img = torch.sparse_coo_tensor(indices, values, size=img_size, dtype=torch.float32, check_invariants=True).coalesce()
      mask = torch.sparse_coo_tensor(indices, ones, size=mask_size, dtype=torch.float32, check_invariants=True).coalesce()
      imgs.append(img)
      masks.append(mask)
    return imgs, masks

  def octree_to_message(self, imgs, masks):
    dims = 3 if self.is_3d else 2
    octree_depth = len(imgs)

    if octree_depth == 0:
      rospy.logfatal('nbv_3d_cnn_octree_predict: octree_to_message: got octree with zero levels.')
      exit(1)

    message = nbv_3d_cnn_octree_msgs.Octree()
    message.levels = []

    num_channels = None

    for i in range(0, octree_depth):
      img = imgs[i]
      if not img.is_coalesced():
        img = img.coalesce()

      values = img.values().cpu().numpy()
      maybe_num_channels = int(values.shape[-1])
      if i > 0 and values.shape[-1] != num_channels:
        rospy.logfatal('nbv_3d_cnn_octree_predict: octree_to_message: mismatch in number of channels at level %d, expected %d, got %d.' %
                       (i, num_channels, maybe_num_channels))
        exit(1)
      num_channels = maybe_num_channels
      indices = img.indices().cpu().numpy()
      indices = indices[1:] # remove batch dimension
      indices = indices.transpose(1, 0)
      size = list(img.shape)
      size = size[1:][:-1] # remove batch and channel

      level = nbv_3d_cnn_octree_msgs.OctreeLevel()
      level.indices = list(indices.flatten())
      level.values = list(values.flatten())
      level.size = size

      message.levels.append(level)
      pass

    message.num_channels = num_channels

    return message

  def on_predict_octree(self, goal):
    self.on_predict(goal, True)
    pass

  def on_predict_image(self, goal):
    self.on_predict(goal, False)
    pass

  def on_predict(self, goal, is_octree):
    with torch.device("cuda"): # for some reason, torch.set_default_device('cuda') is not working here
      rospy.loginfo('nbv_3d_cnn_octree_predict: action start.')

      if not self.model:
        self.model = self.init_model()
        self.model.eval()
      if is_octree:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

      if is_octree:
        inputs, masks = self.octree_from_message(goal.empty_and_frontier)
        if inputs is None:
          rospy.logerr('nbv_3d_cnn_octree_predict: could not decode empty_and_frontier octree.')
          self.action_server.set_aborted()
          return

        _, uninteresting_masks = self.octree_from_message(goal.uninteresting_octree)
        has_uninteresting_masks = len(uninteresting_masks) > 0
        if has_uninteresting_masks:
          rospy.loginfo('nbv_3d_cnn_octree_predict: predicting with uninteresting masks.')
        else:
          uninteresting_masks = None

      if not is_octree:
        sizes = goal.sizes
        if len(sizes) != self.dims:
          rospy.logerr('nbv_3d_cnn_octree_predict: expected %dD image, received %dD.' % (self.dims, len(sizes)))
          self.action_server.set_aborted()
          return
        if int(np.prod(sizes)) != len(goal.frontier) or int(np.prod(sizes)) != len(goal.empty) or int(np.prod(sizes)) != len(goal.uninteresting):
          rospy.logerr('nbv_3d_cnn_octree_predict: expected %d values, received %d.' % (int(np.prod(sizes)*2), len(goal.empty_and_frontier)))
          self.action_server.set_aborted()
          return
        empty_input = np.asarray(goal.empty, dtype=np.float32)
        empty_input = np.reshape(empty_input, [1, *sizes])
        frontier_input = np.asarray(goal.frontier, dtype=np.float32)
        frontier_input = np.reshape(frontier_input, [1, *sizes])
        input = np.stack([empty_input, frontier_input], axis=1) # channel is second axis for pytorch
        input = torch.tensor(input, dtype=torch.float32)

      rospy.loginfo('nbv_3d_cnn_octree_predict: predicting.')
      time_before_prediction = time.perf_counter()
      with torch.no_grad():
        if is_octree:
          outputs, output_masks, _, _ = self.model(inputs, masks, forced_masks=None, uninteresting_masks=uninteresting_masks)
          [output.cpu() for output in outputs] # ensure end of processing
          total_output_values = np.sum([om.values().shape[0] for om in output_masks])
        if not is_octree:
          output = self.model(input)
          output.cpu() # ensure end of processing
          total_output_values = np.prod(list(output.shape))
      prediction_time = time.perf_counter() - time_before_prediction

      rospy.loginfo('nbv_3d_cnn_predict: prediction in %f s.' % float(prediction_time))

      rospy.loginfo('nbv_3d_cnn_predict: sending result.')

      if is_octree:
        result = nbv_3d_cnn_octree_msgs.PredictOctreeResult()
        result.octree_scores = self.octree_to_message(outputs, output_masks)
      if not is_octree:
        result = nbv_3d_cnn_octree_msgs.PredictImageResult()
        result.image_scores = output.cpu().numpy().flatten()

      result.prediction_time = float(prediction_time)
      result.total_output_values = int(total_output_values)
      result.memory_allocated = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

      self.action_server.set_succeeded(result)

      rospy.loginfo('nbv_3d_cnn_predict: action succeeded.')
      pass
    pass

rospy.init_node('nbv_3d_cnn_octree_predict', anonymous=True)

if torch.cuda.is_available():
  rospy.loginfo("CUDA is available")
else:
  rospy.logfatal("CUDA is NOT available")
  exit(1)

rospy.loginfo("CUDA total memory: %d MB" % (torch.cuda.get_device_properties(0).total_memory / (1024*1024)))

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
max_memory_fraction = min(float(max_memory_mb) * 1024*1024 / float(torch.cuda.get_device_properties(0).total_memory), 1.0)
rospy.loginfo("CUDA max memory fraction %f (%d MB)" % (max_memory_fraction, max_memory_mb))
torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
torch.set_default_device('cuda')

server = PredictAction()
rospy.spin()
