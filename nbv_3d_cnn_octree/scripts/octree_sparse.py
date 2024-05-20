#!/usr/bin/python3

import nbv_3d_cnn_msgs.msg as nbv_3d_cnn_msgs
import std_msgs.msg as std_msgs
from rospy.numpy_msg import numpy_msg

from dataset import DatasetFactory, NormalizePoints, PointDatasetWithoutGT, PointDatasetWithGT

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append(os.path.dirname(__file__) + "/o-cnn/tensorflow/script/")
sys.path.append(os.path.dirname(__file__) + "/o-cnn/tensorflow/")
import datetime

import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

import numpy as np
import math
import pickle

import rospy
import actionlib
import cv_bridge

class SparseConv2DLayer(tf.keras.layers.Layer):

  def __init__(self, nout, size=3, name="sparse_conv2d"):
    super().__init__(name = tf.get_current_name_scope() + "/" + name)
    self.nout = nout
    self.size = size

  def build(self, input_shape):
    init_weights = tf.keras.initializers.GlorotNormal()
    self.w = self.add_weight(initializer=init_weights, shape=[self.size, self.size, self.nout],
                             name='weights', trainable=True, dtype=tf.float32)

    init_bias = tf.keras.initializers.GlorotNormal()
    self.b = self.add_weight(initializer=init_bias, shape=[self.nout],
                             name='bias', trainable=True, dtype=tf.float32)

  def call(self, input, mask):
    result = tf.sparse.from_dense(tf.zeros(tf.shape(input), dtype=tf.float32))
    for kx in range(0, self.size):
      for ky in range(0, self.size):
        k = tf.slice(self.w, [ky, kx, 0], [1, 1, self.nout])
        indices = input.indices + tf.constant([ky - self.size // 2, kx - self.size // 2], dtype=tf.int64)
        values = tf.reshape(input.values * k, tf.shape(input.values))
        temp = tf.sparse.SparseTensor(indices, values, input.dense_shape)
        result = tf.sparse.add(result, temp)

    shape_mask = tf.math.logical_and(tf.math.less(result.indices, input.dense_shape),
                               tf.math.greater_equal(result.indices, tf.constant(0, dtype=tf.int64)))
    shape_mask = tf.math.reduce_all(shape_mask, axis=1)
    indices = tf.boolean_mask(result.indices, shape_mask)
    values = tf.boolean_mask(result.values, shape_mask)

    print("mask indices: " + str(mask.indices))
    print("indices: " + str(indices))

    result = tf.sparse.SparseTensor(indices, values, input.dense_shape)
    print("before bias add: " + str(result))
    result = tf.sparse.add(result, mask * self.b)
    print("after bias add: " + str(result))
    return result

  pass

class Trainer:
  def __init__(self):
    self.model = None

    self.checkpoint_file = rospy.get_param('~checkpoint_file', '')
    self.model_type = rospy.get_param('~model_type', '')

    self.ocnn_dataset_root = rospy.get_param('~ocnn_dataset_root', '')

    self.log_file_prefix = rospy.get_param('~log_file_prefix', '')

    self.tensorboard_dir = rospy.get_param('~tensorboard_dir', '')

    self.checkpoint_prefix = rospy.get_param('~checkpoint_prefix', '')

    self.checkpoint_every_iter = rospy.get_param('~checkpoint_every_iter', '')
    pass

  def sparse_conv2d(self, input, size):
    kernel = tf.constant([[0, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0],
                         ], dtype=tf.float32)

    result = tf.sparse.from_dense(tf.zeros(tf.shape(input), dtype=tf.float32))
    for kx in range(0, size):
      for ky in range(0, size):
        k = tf.slice(kernel, [ky, kx], [1, 1])
        indices = input.indices + tf.constant([ky - size // 2, kx - size // 2], dtype=tf.int64)
        values = tf.reshape(input.values * k, tf.shape(input.values))
        mask = tf.math.logical_and(tf.math.less(indices, input.dense_shape), tf.math.greater(indices, tf.constant(0, dtype=tf.int64)))
        mask = tf.math.reduce_all(mask, axis=1)
        indices = tf.boolean_mask(indices, mask)
        values = tf.boolean_mask(values, mask)
        temp = tf.sparse.SparseTensor(indices, values, input.dense_shape)
        result = tf.sparse.add(result, temp)
    return result

  def sparse_octree_test(self):
    rospy.loginfo('sparse_octree_test: start.')

    incomplete_example = tf.constant([[0, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 1, 0, 1],
                                      [0, 0, 0, 1],
                                     ], dtype=tf.float32)
    print("incomplete example: " + str(incomplete_example))

    incomplete_mask = tf.constant([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 1, 1],
                                   [0, 0, 1, 1],
                                  ], dtype=tf.float32)
    print("incomplete mask: " + str(incomplete_mask))

    complete_example = tf.constant([[0, 1, 0, 1],
                                    [0, 1, 0, 1],
                                    [0, 1, 0, 1],
                                    [0, 1, 0, 1],
                                   ], dtype=tf.float32)
    print("complete example: " + str(complete_example))

    incomplete_sparse = tf.sparse.from_dense(incomplete_example)
    incomplete_mask_sparse = tf.sparse.from_dense(incomplete_example)
    print("incomplete sparse: " + str(incomplete_sparse))

    #convoluted_sparse = self.sparse_conv2d(incomplete_sparse, 3)
    layer = SparseConv2DLayer(1)
    convoluted_sparse = SparseConv2DLayer(1)(incomplete_sparse, incomplete_mask_sparse)
    #print("convoluted_sparse: " + str(convoluted_sparse))
    print("convoluted_sparse: " + str(tf.sparse.to_dense(convoluted_sparse)))

    pass

rospy.init_node('octree_sparse', anonymous=True)

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
# limit GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
      gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory_mb)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print("Exception while limiting GPU memory:")
    print(e)
    exit()

trainer = Trainer()
trainer.sparse_octree_test();
