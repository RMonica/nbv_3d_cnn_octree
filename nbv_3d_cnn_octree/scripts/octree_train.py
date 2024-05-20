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

from network_completion import CompletionResnet
from config import FLAGS
from libs import octree_scan, octree_batch, normalize_points, points2octree, transform_points

# the dataset

class LearningRateGen:
  def __init__(self, flags):
    step_size = list(flags.step_size)
    for i in range(len(step_size), 5):
      step_size.append(step_size[-1])

    self.steps = step_size
    for i in range(1, 5):
      self.steps[i] = self.steps[i-1] + self.steps[i]
    self.lr_values = [flags.gamma**i * flags.learning_rate for i in range(0, 6)]

    print("steps: " + str(self.steps))
    print("lr_values: " + str(self.lr_values))
    pass

  def get_learning_rate(self, iteration):
    for i in range(0, len(self.steps)):
      if iteration < self.steps[i]:
        return self.lr_values[i]
    pass

  pass

class Trainer:
  def __init__(self):
    self.model = None

    global FLAGS
    FLAGS.DATA.train.camera = '_' # used to generate partial scans
    FLAGS.MODEL.skip_connections = True
    config_file = rospy.get_param('~config', "")
    if (config_file != ""):
      FLAGS.merge_from_file(config_file)

    self.weight_decay = FLAGS.LOSS.weight_decay

    self.checkpoint_file = rospy.get_param('~checkpoint_file', '')
    self.model_type = rospy.get_param('~model_type', '')

    self.ocnn_dataset_root = rospy.get_param('~ocnn_dataset_root', '')

    self.log_file_prefix = rospy.get_param('~log_file_prefix', '')

    self.tensorboard_dir = rospy.get_param('~tensorboard_dir', '')

    self.checkpoint_prefix = rospy.get_param('~checkpoint_prefix', '')

    self.checkpoint_every_iter = rospy.get_param('~checkpoint_every_iter', '')

    FLAGS.DATA.train.location = self.ocnn_dataset_root + FLAGS.DATA.train.location
    FLAGS.DATA.test.location = self.ocnn_dataset_root + FLAGS.DATA.test.location
    FLAGS.DATA.train.camera = self.ocnn_dataset_root + FLAGS.DATA.train.camera
    pass

  def train(self):
    rospy.loginfo('octree_train: training start.')

    global FLAGS

    network = CompletionResnet(FLAGS.MODEL)

    octree_in = tf.keras.layers.Input(shape=(None, 1), name="octree_in", dtype=tf.string)
    octree_gt = tf.keras.layers.Input(shape=(None, 1), name="octree_gt", dtype=tf.string)
    #batched_octree_in = octree_batch([octree_in, ])
    #batched_octree_gt = octree_batch([octree_gt, ])
    batched_octree_in = tf.keras.layers.Input(shape=(), name="batched_octree_in", dtype=tf.int8)
    batched_octree_gt = tf.keras.layers.Input(shape=(), name="batched_octree_gt", dtype=tf.int8)

    convd = network.octree_encoder(batched_octree_in, training=True, reuse=False)
    #batched_octree_out = network.decode_shape(convd, batched_octree_in, training=True, reuse=False)
    loss, accu = network.octree_decoder(convd, batched_octree_in, batched_octree_gt, training=True, reuse=False)

    total_loss = tf.add_n(loss)

    model = tf.keras.models.Model([batched_octree_in, batched_octree_gt], total_loss, name='octree_complete_training_model')

    model.compile(optimizer='sgd', loss='mse')

    #model.summary()
    if (self.checkpoint_file != ""):
      model.load_weights(self.checkpoint_file)

    log_file_name = ""
    if self.log_file_prefix != "":
      log_file_name = self.log_file_prefix + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "-") + ".csv"
      rospy.loginfo("octree_train: logging to file: " + log_file_name)

    dataset = DatasetFactory(FLAGS.DATA.train, NormalizePoints, PointDatasetWithGT)()

    current_learning_rate = 0.1
    get_current_learning_rate = lambda: current_learning_rate

    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=get_current_learning_rate, momentum=0.9, name='SGD')

    batch_size = FLAGS.DATA.train.batch_size
    max_iter = FLAGS.SOLVER.max_iter
    iteration = 0

    learning_rate_function = LearningRateGen(FLAGS.SOLVER)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    trainable_weights = model.trainable_weights
    # get all weights named "weights" (exclude biases)
    weights_trainable_weights = [w for w in model.trainable_weights if ('/weights:' in w.name)]

    if log_file_name != "":
      logfile = open(log_file_name, "w")
      logfile.write("time, iteration, model_loss, l2_regularization_loss, total_loss\n")
      logfile.close()

    start_ros_time = rospy.get_time()
    start_ros_time_after_first_iteration = rospy.get_time()

    summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)

    for s in dataset:
      learning_rate = learning_rate_function.get_learning_rate(iteration)

      with tf.GradientTape() as tape:
        model_loss = model([s[0], s[1]])

        #print("weight: ")
        #print([(w.name, float(tf.nn.l2_loss(w))) for w in weights_trainable_weights])

        l2_regularization_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights_trainable_weights]) * self.weight_decay

        loss_value = model_loss + l2_regularization_loss

        rospy.loginfo("octree_train: ---- %d / %d (%02.2f%%)----" % (int(iteration), int(max_iter), float(iteration / max_iter * 100)))
        rospy.loginfo("octree_train: model_loss: " + str(float(model_loss)))
        rospy.loginfo("octree_train: l2_regularization_loss: " + str(float(l2_regularization_loss)))
        rospy.loginfo("octree_train: loss_value: " + str(float(loss_value)))
        pass

      with summary_writer.as_default():
        tf.summary.scalar("model_loss", model_loss, step=iteration)
        tf.summary.scalar("l2_regularization_loss", l2_regularization_loss, step=iteration)
        tf.summary.scalar("train_loss", loss_value, step=iteration)
        pass

      gradients = tape.gradient(loss_value, model.trainable_weights)

      sum_gradients = sum([float(tf.nn.l2_loss(g)) for g in gradients])
      if math.isnan(sum_gradients) or (sum_gradients > 10000.0):
        rospy.logwarn("octree_train: sum of gradients is %f (likely corrupted), skipping iteration %d" % (sum_gradients, iteration))
        continue

      current_ros_time = rospy.get_time()
      rospy.loginfo("octree_train: ETA: %f hours" % ((current_ros_time - start_ros_time_after_first_iteration) /
                                                    max(iteration, 1) * (max_iter - iteration) / 3600.0))

      if log_file_name != "":
        logfile = open(log_file_name, "a")
        logfile.write("%f, %d, %f, %f, %f\n" % (float(current_ros_time - start_ros_time), iteration,
                                                float(model_loss), float(l2_regularization_loss), float(loss_value)))
        logfile.close()

      sgd_optimizer.apply_gradients(zip(gradients, model.trainable_weights))
      #print(s[0])
      #print("s shape: " + str(s[0].shape))
      #result = model.predict([s[0], s[1]], batch_size=(s[0].shape[0]))
      #print(result)
      #print("result length: " + str(len(result)))
      iteration += 1

      if iteration == 1:
        start_ros_time_after_first_iteration = rospy.get_time()

      # SAVE CHECKPOINT
      if (iteration % self.checkpoint_every_iter == 0) or (iteration > max_iter):
        filename = self.checkpoint_prefix + str(iteration)
        rospy.loginfo("octree_train: saving checkpoint %s" % filename)
        model.save_weights(filename)

      if iteration > max_iter:
        break
      if rospy.is_shutdown():
        break

    pass
  pass

rospy.init_node('octree_train', anonymous=True)

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
trainer.train();
