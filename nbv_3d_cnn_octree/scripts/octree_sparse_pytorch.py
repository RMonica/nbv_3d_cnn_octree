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
import struct

import rospy
import actionlib
import cv_bridge
import cv2

import torch

from octree_sparse_pytorch_common import SparseLogitsLoss
from octree_sparse_pytorch_common import SparseOutputLoss
from octree_sparse_pytorch_common import SparseUnifiedOutputLoss

import model_factory
import octree_save_load
import octree_sparse_pytorch_utils
from enum_config import ModelType, InputType, EngineType

class Trainer:

  def __init__(self):
    self.ModelType = ModelType
    self.InputType = InputType
    self.EngineType = EngineType

    self.model = None

    self.load_checkpoint_file = rospy.get_param('~load_checkpoint_file', '')

    self.first_train_image = rospy.get_param('~first_train_image', 0)
    self.last_train_image = rospy.get_param('~last_train_image', 120)
    self.first_test_image = rospy.get_param('~first_test_image', 120)
    self.last_test_image = rospy.get_param('~last_test_image', 180)

    self.ocnn_dataset_root = rospy.get_param('~ocnn_dataset_root', '')

    self.weight_decay = rospy.get_param("~weight_decay", 0.0)

    self.input_prefix = rospy.get_param('~input_prefix', '')
    self.test_prefix = rospy.get_param('~test_prefix', '')

    self.max_levels = rospy.get_param('~max_levels', 3)

    self.save_intermediate_tests = rospy.get_param('~save_intermediate_tests', True)
    self.save_last_test = rospy.get_param('~save_last_test', True)

    self.label_hidden_channels = rospy.get_param("~label_hidden_channels", 16)

    self.resblock_num = rospy.get_param('~resblock_num', 3)

    self.num_epochs = rospy.get_param('~num_epochs', 120)

    self.batch_size = rospy.get_param('~batch_size', 1)

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
    try:
      self.crop_image_x = self.crop_image[0]
      self.crop_image_y = self.crop_image[1]
      if self.is_3d:
        self.crop_image_z = self.crop_image[2]
    except ValueError:
      rospy.logfatal("Could not convert %s to integers." % str(self.crop_image))
      exit(1)
    rospy.loginfo("image will be cropped at %s" % str(self.crop_image))

    self.model_type = rospy.get_param('~model_type', self.ModelType.SPARSE_RESNET)
    self.model_type = self.ModelType(self.model_type)

    if self.model_type == self.ModelType.SPARSE_RESNET:
      self.input_type = self.InputType.OCTREE
    elif self.model_type == self.ModelType.SPARSE_ENC_DEC:
      self.input_type = self.InputType.OCTREE
    elif self.model_type == self.ModelType.RESNET:
      self.input_type = self.InputType.IMAGE
    elif self.model_type == self.ModelType.ENC_DEC:
      self.input_type = self.InputType.IMAGE
    else:
      rospy.logfatal("Unknown model_type parameter: " + self.model_type)
      exit(1)

    # learning rate at first epoch
    self.learning_rate = rospy.get_param('~learning_rate', 0.01)
    # learning rate at last epoch
    self.last_learning_rate = rospy.get_param('~last_learning_rate', self.learning_rate)

    self.base_channels = rospy.get_param('~base_channels', 16)
    self.max_channels = rospy.get_param('~max_channels', 1000000000)

    self.training_with_unified_loss = rospy.get_param('~training_with_unified_loss', False)
    self.enable_unified_loss_at_epoch = rospy.get_param('~enable_unified_loss_at_epoch', 0)

    self.max_allowed_gradient = rospy.get_param('~max_allowed_gradient', 0.1)
    self.max_allowed_gradient = self.max_allowed_gradient / self.learning_rate

    self.enable_gradient_clipping = rospy.get_param('~enable_gradient_clipping', False)

    self.logit_leak = rospy.get_param('~logit_leak', 0.5)

    self.initial_epoch = rospy.get_param('~initial_epoch', 0)

    self.engine = rospy.get_param("~engine", self.EngineType.PYTORCH)
    self.engine = self.EngineType(self.engine)

    self.tensorboard_dir = rospy.get_param('~tensorboard_dir', '')

    self.checkpoint_prefix = rospy.get_param('~checkpoint_prefix', '')

    self.test_every_iter = rospy.get_param('~test_every_iter', '')

    self.unified_loss_alpha = rospy.get_param('~unified_loss_alpha', '')

    self.mse_leak = rospy.get_param('~mse_leak', 0.01)

    self.now = datetime.datetime.now()
    pass

  def get_stats_filename(self):
    date_time = self.now.strftime("%Y_%m_%d_%H_%M_%S_")
    return self.test_prefix + date_time + self.model_type.value + "_mse_stats.csv";

  def get_train_stats_filename(self):
    date_time = self.now.strftime("%Y_%m_%d_%H_%M_%S_")
    return self.checkpoint_prefix + date_time + self.model_type.value + "_train_stats.csv";

  def get_test_stats_filename(self):
    date_time = self.now.strftime("%Y_%m_%d_%H_%M_%S_")
    return self.checkpoint_prefix + date_time + self.model_type.value + "_test_stats.csv";

  def reset_stats_file(self):
    # clear all files
    open(self.get_stats_filename(), "w")
    train_stats_file = open(self.get_train_stats_filename(), "w")
    train_stats_file.write("Iter; Time; Mem; PredTime; BackpTime; LearningRate; Loss; AVGTotalOutputValues; PartialLosses \n")
    test_stats_file = open(self.get_test_stats_filename(), "w")
    test_stats_file.write("Iter; Mem; PredTime; Loss; AVGTotalOutputValues \n")
    pass

  def save_train_stats(self, iter, time, avg_mem, avg_pred_time, avg_backp_time, avg_loss, learning_rate, avg_total_output_values,
                       partial_losses):
    train_stats_file = open(self.get_train_stats_filename(), "a")
    train_stats_file.write("%d; %f; %f; %f; %f; %f; %f; %f; %s\n" %
                           (iter, time, avg_mem, avg_pred_time, avg_backp_time, learning_rate, avg_loss, avg_total_output_values,
                            ("%f|%f|%f" % tuple(partial_losses))))

  def save_test_stats(self, iter, avg_mem, avg_pred_time, avg_loss, avg_total_output_values):
    test_stats_file = open(self.get_test_stats_filename(), "a")
    test_stats_file.write("%s; %f; %f; %f; %f\n" % (str(iter), avg_mem, avg_pred_time, avg_loss, avg_total_output_values))

  def dilate_3d_numpy(self, image, min=0, max=1):
    shape = image.shape
    shifted_xp = image.copy()
    shifted_xm = image.copy()
    shifted_yp = image.copy()
    shifted_ym = image.copy()
    shifted_zp = image.copy()
    shifted_zm = image.copy()

    for i in range(1, shape[0]):
      shifted_zm[i - 1, :, :] = image[i, :, :]
    for i in range(0, shape[0]-1):
      shifted_zp[i + 1, :, :] = image[i, :, :]

    for i in range(1, shape[1]):
      shifted_ym[:, i - 1, :] = image[:, i, :]
    for i in range(0, shape[1]-1):
      shifted_yp[:, i + 1, :] = image[:, i, :]

    for i in range(1, shape[2]):
      shifted_xm[:, :, i - 1] = image[:, :, i]
    for i in range(0, shape[2]-1):
      shifted_xp[:, :, i + 1] = image[:, :, i]

    result = image + shifted_xm + shifted_ym + shifted_zm + shifted_xp + shifted_yp + shifted_zp
    result = result.clip(min, max)
    return result

  def test_and_save_images(self, model, iter, is_last_test=False):
    stats_filename = self.get_stats_filename();
    stats_file = open(stats_filename, "a")
    avg_mse = 0.0
    tot_img = 0
    stats_file.write(str(iter) + "; ")

    model.eval()

    avg_mem       = 0
    avg_pred_time = 0
    avg_loss      = 0
    avg_counter   = 0
    avg_total_output_values = 0

    with torch.no_grad():
      for image_num in range(self.first_test_image, self.last_test_image):
        rospy.loginfo("TESTING IMAGE %d" % image_num)
        torch.cuda.empty_cache()

        gt_octree_filename = self.input_prefix + str(image_num) + "_gt_octree.octree";
        input_octree_filename = self.input_prefix + str(image_num) + "_input_octree.octree";
        occupied_octree_filename = self.input_prefix + str(image_num) + "_occupied_octree.octree";
        with open(gt_octree_filename, "rb") as ifile:
          gt_outputs, gt_masks, _, _, _, _ = octree_save_load.load_octree_from_file(ifile)
        with open(input_octree_filename, "rb") as ifile:
          inputs, masks, uninteresting_masks, _, _, _ = octree_save_load.load_octree_from_file(ifile)
          empty_inputs = [octree_save_load.sparse_select(inp, index=0, remove_dim=False) for inp in inputs]
        with open(occupied_octree_filename, "rb") as ifile:
          occupied_inputs, occupied_masks, _, _, _, _ = octree_save_load.load_octree_from_file(ifile)

        if self.input_type == self.InputType.IMAGE:
          gt_output = octree_save_load.octree_to_pytorch_image(gt_outputs, gt_masks, self.crop_image).unsqueeze(0)
          input = octree_save_load.octree_to_pytorch_image(inputs, masks, self.crop_image).unsqueeze(0)
          gt_mask = octree_save_load.octree_to_pytorch_image(gt_masks, gt_masks, self.crop_image).unsqueeze(0)

        rospy.loginfo('nbv_3d_cnn_octree_predict: predicting.')
        time_before_prediction = time.perf_counter()
        if self.input_type == self.InputType.OCTREE:
          outputs, output_masks, output_logits, _ = model(inputs, masks, forced_masks=None, uninteresting_masks=uninteresting_masks)
          [output.cpu() for output in outputs] # ensure end of processing
          total_output_values = np.sum([om.values().shape[0] for om in output_masks])
        if self.input_type == self.InputType.IMAGE:
          output = model(input)
          output.cpu() # ensure end of processing
          total_output_values = 0.0
        time_after_prediction = time.perf_counter()
        prediction_time = time.perf_counter() - time_before_prediction
        rospy.loginfo('nbv_3d_cnn_octree_predict: predicted in %f.' % prediction_time)

        output_logits = None

        reconst_uninteresting_mask = octree_save_load.octree_to_image(uninteresting_masks, uninteresting_masks)
        reconst_uninteresting_mask = octree_save_load.image_crop(reconst_uninteresting_mask, self.crop_image)
        reconst_interesting_mask = octree_save_load.octree_to_image(gt_masks, gt_masks)
        reconst_interesting_mask = octree_save_load.image_crop(reconst_interesting_mask, self.crop_image)
        reconst_gt = octree_save_load.octree_to_image(gt_outputs, gt_masks)
        reconst_gt = octree_save_load.image_crop(reconst_gt, self.crop_image)
        reconst_gt = reconst_gt * reconst_interesting_mask
        reconst_inputs = octree_save_load.octree_to_image(inputs, masks)
        reconst_inputs = octree_save_load.image_crop(reconst_inputs, self.crop_image)

        if self.input_type == self.InputType.OCTREE:
          reconst_image = octree_save_load.octree_to_image(outputs, output_masks)
          reconst_image = octree_save_load.image_crop(reconst_image, self.crop_image)
          reconst_image = reconst_image * reconst_interesting_mask
        if self.input_type == self.InputType.IMAGE:
          reconst_image = output.numpy(force=True)[0]
          reconst_image = octree_save_load.image_channels_last(reconst_image)
          reconst_image = reconst_image * reconst_interesting_mask

        mse = math.sqrt(np.square(reconst_gt - reconst_image).mean());
        rospy.loginfo("  RMSE %f" % mse)
        stats_file.write(str(mse) + "; ")
        avg_mse += mse
        tot_img += 1

        reconst_empty = octree_save_load.select_nth_channel(reconst_inputs, 0)
        if not self.is_3d:
          kernel = np.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], np.uint8)
          reconst_empty_dilated = cv2.dilate(reconst_empty, kernel, iterations = 1)
        else:
          reconst_empty_dilated = self.dilate_3d_numpy(reconst_empty)

        reconst_occupied = octree_save_load.select_nth_channel(reconst_inputs, 1)

#        if self.input_type == self.InputType.OCTREE:
#          output_test_filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_output_test.octree";
#          with open(output_test_filename, "wb") as ofile:
#            octree_save_load.save_octree_to_file(ofile, outputs, output_masks)

        if self.save_intermediate_tests or (is_last_test and self.save_last_test):
          if not self.is_3d:
            output_test_filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) + "_output_test.png";
            output_test_image = (reconst_image.squeeze(-1) + reconst_occupied) * 255
            cv2.imwrite(output_test_filename, output_test_image)
            output_test_filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_gt_test.png";
            cv2.imwrite(output_test_filename, reconst_gt.squeeze(-1) * 255)

            output_test_filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_merged_test.png";
            merged_image = np.concatenate([reconst_image, reconst_image, reconst_image], axis=-1)
            #uninteresting_reconst_gt = reconst_gt * reconst_uninteresting_mask
            merged_image[:, :, 2] = (reconst_uninteresting_mask)[:, :, 0]
            merged_image[:, :, 1] = reconst_occupied[:, :]
            merged_image[:, :, 0] = (reconst_image * reconst_interesting_mask)[:, :, 0]
            cv2.imwrite(output_test_filename, merged_image * 255)
            pass # not 3d

          if self.is_3d and self.input_type == self.InputType.IMAGE:
            filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) + "_output_test"
            output_test_image = reconst_image.squeeze(-1) + (reconst_occupied * 1.1)
            octree_save_load.save_voxelgrid(filename, output_test_image)

          if self.input_type == self.InputType.OCTREE:
            filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) + "_output_merged.octree"
            output_test_octree, output_test_masks = outputs, output_masks
            s_empty, s_masks = octree_sparse_pytorch_utils.scale_values_octree(empty_inputs, masks, scale=1.1)
            output_test_octree, output_test_masks = octree_sparse_pytorch_utils.merge_octrees(output_test_octree, output_test_masks,
                                                                                              s_empty, s_masks, is_3d=self.is_3d,
                                                                                              operation="subtract_and_clamp_01",
                                                                                              clamp_min=-0.1, clamp_max=1.1)
            s_occupied, s_omasks = octree_sparse_pytorch_utils.scale_values_octree(occupied_inputs, occupied_masks, scale=1.1)
            output_test_octree, output_test_masks = octree_sparse_pytorch_utils.merge_octrees(output_test_octree, output_test_masks,
                                                                                              s_occupied, s_omasks, is_3d=self.is_3d,
                                                                                              operation="add_and_clamp_01",
                                                                                              clamp_min=-0.1, clamp_max=1.1)
            s_empty = None; s_masks = None; s_occupied = None; s_omasks = None
            with open(filename, "wb") as ofile:
              octree_save_load.save_octree_to_file(ofile, output_test_octree, output_test_masks)
  #          filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_gt_test";
  #          octree_save_load.save_voxelgrid(filename, reconst_gt.squeeze(-1))

  #          filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_uninteresting";
  #          octree_save_load.save_voxelgrid(filename, reconst_uninteresting_mask.squeeze(-1))
  #          filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_occupied";
  #          octree_save_load.save_voxelgrid(filename, reconst_occupied)
  #          filename = self.test_prefix + self.model_type.value + "_" + str(iter) + "_" + str(image_num) +  "_predict";
  #          octree_save_load.save_voxelgrid(filename, (reconst_image * reconst_interesting_mask).squeeze(-1))
            pass # is 3d
          pass # self.save_intermediate_tests or (is_last_test and self.save_last_test)

        pred_time = time_after_prediction - time_before_prediction
        rospy.loginfo("prediction: %fs" % pred_time)
        display_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        rospy.loginfo("model memory MB: " + str(display_mem))

        avg_mem += display_mem
        avg_pred_time += pred_time
        avg_loss += mse
        avg_total_output_values += total_output_values
        avg_counter += 1

        # cleanup memory
        outputs = None
        output_masks = None
        output_logits = None

        if rospy.is_shutdown():
          return
        pass
      pass

    avg_mse /= tot_img
    stats_file.write(str(avg_mse) + "\n")

    avg_mem /= avg_counter
    avg_pred_time /= avg_counter
    avg_loss /= avg_counter
    avg_total_output_values /= avg_counter
    self.save_test_stats(iter, avg_mem=avg_mem, avg_pred_time=avg_pred_time, avg_loss=avg_loss,
                         avg_total_output_values=avg_total_output_values)

    # restore training mode
    model.train()
    pass

  def merge_batch(self, batch_list):
    levels = []
    for sample in batch_list:
      for l, level in enumerate(sample):
        while len(levels) < l + 1:
          levels.append([])
        levels[l].append(level)
        pass
      pass

    result = [(torch.cat(i).coalesce() if torch.cat(i).is_sparse else torch.stack(i)) for i in levels]
    return result


  def sparse_octree_test(self):
    rospy.loginfo('sparse_octree_test: start.')

    octree_depth = self.max_levels

    #resblock_num = 3
    resblock_num = self.resblock_num
    model = None
    rospy.loginfo("sparse_octree_test: Engine is %s" % str(self.engine))
    model = model_factory.get_model(model_type=self.model_type, engine=self.engine,
                                    nin=2, nout=1, octree_depth=octree_depth, resblock_num=self.resblock_num, max_channels=self.max_channels,
                                    base_channels=self.base_channels, label_hidden_channels=self.label_hidden_channels,
                                    is_3d=self.is_3d)

    if model is None:
      rospy.logfatal("Model not implemented with the given configuration.")
      exit(1)

    rospy.loginfo("Model parameters are: ")
    for name, p in model.named_parameters():
      rospy.loginfo("  " + name)

    if self.input_type == self.InputType.OCTREE:
      logits_loss = SparseLogitsLoss()
      output_loss = SparseOutputLoss(dims=self.dims)
      unified_loss = SparseUnifiedOutputLoss(dims=self.dims, unified_alpha=self.unified_loss_alpha, mse_leak=self.mse_leak)
    if self.input_type == self.InputType.IMAGE:
      image_loss = torch.nn.MSELoss()

    if self.load_checkpoint_file != "":
      model.load_state_dict(torch.load(self.load_checkpoint_file))

    #optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    #torch.autograd.set_detect_anomaly(True)

    self.reset_stats_file()
    start_time = time.perf_counter()

    test_every_iter = self.test_every_iter

    batch_accu_counter = 0
    if self.input_type == self.InputType.OCTREE:
      batch_accu_gt_outputs = []
      batch_accu_gt_masks = []
      batch_accu_inputs = []
      batch_accu_masks = []
      batch_accu_uninteresting_masks = []
      batch_accu_image_pyramids = [[], [], []]
      batch_accu_sq_image_pyramids = []
    if self.input_type == self.InputType.IMAGE:
      batch_accu_gt_output = []
      batch_accu_input = []
      batch_accu_gt_mask = []

    lr_scheduler_gamma = 1.0
    if self.num_epochs > 0:
      lr_scheduler_gamma = math.exp(math.log(self.last_learning_rate / self.learning_rate) / self.num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_scheduler_gamma)

    for iter in range(self.initial_epoch, self.num_epochs):

      avg_mem = 0.0
      avg_pred_time = 0.0
      avg_backp_time = 0.0
      avg_loss = 0.0
      avg_partial_losses = [0.0, ] * 3
      avg_partial_losses_counter = [0, ] * 3
      avg_total_output_values = 0.0
      avg_counter = 0

      unified_loss_enabled = (self.training_with_unified_loss and (iter >= self.enable_unified_loss_at_epoch))

      for image_num in range(self.first_train_image, self.last_train_image):
        rospy.loginfo("ITERATION %d IMAGE %d" % (iter, image_num))
        torch.cuda.empty_cache()

        gt_octree_filename = self.input_prefix + str(image_num) + "_gt_octree.octree";
        input_octree_filename = self.input_prefix + str(image_num) + "_input_octree.octree";
        rospy.loginfo("loading gt file " + gt_octree_filename)
        with open(gt_octree_filename, "rb") as ifile:
          gt_outputs, gt_masks, _, _, _, weighted_img_pyramid = octree_save_load.load_octree_from_file(ifile)
        rospy.loginfo("loading input file " + input_octree_filename)
        with open(input_octree_filename, "rb") as ifile:
          inputs, masks, uninteresting_masks, _, _, _ = octree_save_load.load_octree_from_file(ifile)

        if self.input_type == self.InputType.OCTREE:
          batch_accu_gt_outputs.append(gt_outputs)
          batch_accu_gt_masks.append(gt_masks)
          batch_accu_inputs.append(inputs)
          batch_accu_masks.append(masks)
          batch_accu_uninteresting_masks.append(uninteresting_masks)
          for i in range(0, len(batch_accu_image_pyramids)):
            batch_accu_image_pyramids[i].append(weighted_img_pyramid[i])
        if self.input_type == self.InputType.IMAGE:
          gt_output = octree_save_load.octree_to_pytorch_image(gt_outputs, gt_masks, self.crop_image).unsqueeze(0)
          batch_accu_gt_output.append(gt_output)
          input = octree_save_load.octree_to_pytorch_image(inputs, masks, self.crop_image).unsqueeze(0)
          batch_accu_input.append(input)
          gt_mask = octree_save_load.octree_to_pytorch_image(gt_masks, gt_masks, self.crop_image).unsqueeze(0)
          batch_accu_gt_mask.append(gt_mask)
        batch_accu_counter += 1

        if batch_accu_counter >= self.batch_size:
          rospy.loginfo("batch has size %d, start processing." % batch_accu_counter)

          if self.input_type == self.InputType.OCTREE:
            gt_outputs = self.merge_batch(batch_accu_gt_outputs)
            gt_masks = self.merge_batch(batch_accu_gt_masks)
            inputs = self.merge_batch(batch_accu_inputs)
            masks = self.merge_batch(batch_accu_masks)
            uninteresting_masks = self.merge_batch(batch_accu_uninteresting_masks)
            image_pyramids = [self.merge_batch(ba) for ba in batch_accu_image_pyramids]
            batch_accu_gt_outputs = []
            batch_accu_gt_masks = []
            batch_accu_inputs = []
            batch_accu_masks = []
            batch_accu_uninteresting_masks = []
            batch_accu_image_pyramids = [[], [], []]
          if self.input_type == self.InputType.IMAGE:
            gt_output = torch.cat(batch_accu_gt_output)
            input = torch.cat(batch_accu_input)
            gt_mask = torch.cat(batch_accu_gt_mask)
            batch_accu_gt_output = []
            batch_accu_input = []
            batch_accu_gt_mask = []
          batch_accu_counter = 0

          if self.input_type == self.InputType.OCTREE:
            final_mask = octree_save_load.find_final_mask(masks, is_3d=self.is_3d)
            gt_logits = octree_save_load.logits_from_masks(gt_masks, initial_mask=final_mask,
                                                           uninteresting_masks=uninteresting_masks, is_3d=self.is_3d)

          time_before_prediction = time.perf_counter()
          partial_losses = [0.0, ] * 3
          # prediction
          if self.input_type == self.InputType.OCTREE:
            forced_masks = (gt_masks if not unified_loss_enabled else None)
            outputs, output_masks, output_logits, unified_loss_data = model(
              inputs, masks, forced_masks=forced_masks, uninteresting_masks=uninteresting_masks,
              training_with_unified_loss=unified_loss_enabled)
            total_output_values = np.sum([om.values().shape[0] for om in output_masks])
            rospy.loginfo("total output values: %d" % (total_output_values))
            if not unified_loss_enabled:
              l_loss = logits_loss(output_logits, gt_logits)
              rospy.loginfo("current l_loss: " + str(float(l_loss)))
              o_loss = output_loss(outputs, output_masks, image_pyramids, gt_outputs, gt_masks)
              rospy.loginfo("current o_loss: " + str(float(o_loss)))
              loss = l_loss + o_loss
            else:
              rospy.loginfo("current_logit_leak: " + str(self.logit_leak))
              loss = unified_loss(unified_loss_data, image_pyramids, gt_outputs, gt_masks, self.logit_leak)
              partial_losses_c = unified_loss.get_last_partial_losses()
              if not (partial_losses_c is None):
                partial_losses = partial_losses_c
                rospy.loginfo("current partial losses: " + str(partial_losses))
              rospy.loginfo("current unified loss: " + str(float(loss)))
          if self.input_type == self.InputType.IMAGE:
            output = model(input)
            masked_output = output * gt_mask
            masked_gt_output = gt_output * gt_mask
            total_output_values = 0.0
            #print("max output: " + str(torch.max(output).cpu().detach().numpy().squeeze()))
            #print("max gt output: " + str(torch.max(masked_gt_output).cpu().detach().numpy().squeeze()))
            loss = image_loss(masked_output, masked_gt_output)
            rospy.loginfo("current image loss: " + str(loss))
            #loss = torch.mean(torch.pow(masked_output - masked_gt_output, 2.0))

          # weight decay
          if self.weight_decay > 0.0:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            selected_params_list = [torch.sum(torch.square(x)) for name, x in model.named_parameters()
                                                               if (not (".bias" in name)) and
                                                                  (x.grad is not None) and (torch.max(torch.abs(x.grad)) > 0.0)
                                   ]
            if len(selected_params_list) > 0:
              l2_regularization_loss = self.weight_decay * torch.sum(torch.stack(selected_params_list))
              rospy.loginfo("current l2_regularization_loss: " + str(float(l2_regularization_loss)))
              loss = loss + l2_regularization_loss

          time_after_prediction = time.perf_counter()

          display_loss = float(loss.item())
          display_mem = torch.cuda.memory_allocated() / (1024 * 1024)
          rospy.loginfo("current loss: " + str(display_loss))
          rospy.loginfo("model memory MB: " + str(display_mem))

          time_before_backp = time.perf_counter()

          optimizer.zero_grad()
          loss.backward()
          found_high = False
          for name, param in model.named_parameters():
            #print("name: " + name + " max: " + (str(torch.max(param.grad)) if param.grad is not None else "None"))
            #if (param.grad is not None):
            #  print("name: " + name + " (torch.max(torch.abs(param.grad)) > self.max_allowed_gradient): " +
            #        str(float(torch.max(torch.abs(param.grad)))) + " > " + str(self.max_allowed_gradient))
            #print("name: " + name + " max: " + str(float(torch.max(torch.abs(param)))))
            if (param.grad is not None) and (torch.max(torch.abs(param.grad)) > self.max_allowed_gradient):
              found_high = True
              rospy.logerr("Parameter %s gradient is too high (%f)!" % (name, float(torch.max(torch.abs(param.grad)))))
          if found_high and self.enable_gradient_clipping:
            rospy.logwarn("loss too high, enabling gradient clipping.")
            try:
              torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_allowed_gradient, error_if_nonfinite=True, norm_type='inf')
              found_high = False
            except RuntimeError:
              rospy.logerr("Gradient is not finite!")
          if found_high:
            rospy.logwarn("gradient too high or NAN, skipping.")
            optimizer.zero_grad()
            loss = None
            l_loss = None
            o_loss = None
            outputs = None
            output_masks = None
            output_logits = None
            unified_loss_data = None
            rospy.loginfo("model memory MB: " + str(torch.cuda.memory_allocated() / (1024 * 1024)))
            if rospy.is_shutdown():
              break
            continue
          optimizer.step()

          time_after_backp = time.perf_counter()

          rospy.loginfo("prediction: %fs" % (time_after_prediction - time_before_prediction))
          rospy.loginfo("backpropagation: %fs" % (time_after_backp - time_before_backp))

          # update stats
          avg_mem += display_mem
          avg_pred_time += time_after_prediction - time_before_prediction
          avg_backp_time += time_after_backp - time_before_backp
          avg_total_output_values += total_output_values
          avg_loss += display_loss
          avg_partial_losses = [partial_losses[dd] + avg_partial_losses[dd] for dd in range(0, 3)]
          avg_partial_losses_counter = [(avg_partial_losses_counter[dd] + (0 if partial_losses[dd] == 0.0 else 1)) for dd in range(0, 3)]
          avg_counter += 1

          if rospy.is_shutdown():
            break
          pass
        pass

      if rospy.is_shutdown():
        break

      if avg_counter == 0:
        avg_counter = 1 # prevent division-by-zero
      for dd in range(0, 3):
        if avg_partial_losses_counter[dd] == 0:
          avg_partial_losses_counter[dd] = 1

      avg_mem = avg_mem / avg_counter
      avg_pred_time = avg_pred_time / avg_counter
      avg_backp_time = avg_backp_time / avg_counter
      avg_loss = avg_loss / avg_counter
      avg_partial_losses = [avg_partial_losses[dd] / avg_partial_losses_counter[dd] for dd in range(0, 3)]
      avg_total_output_values = avg_total_output_values / avg_counter
      elapsed_time = time.perf_counter() - start_time
      self.save_train_stats(iter, time=elapsed_time, avg_mem=avg_mem,
                            avg_pred_time=avg_pred_time, avg_backp_time=avg_backp_time, avg_loss=avg_loss, partial_losses=avg_partial_losses,
                            learning_rate=(float(lr_scheduler.get_last_lr()[0])), avg_total_output_values=avg_total_output_values)

      if (iter % test_every_iter) == 0:
        self.test_and_save_images(model, iter)
        checkpoint_test_filename = self.checkpoint_prefix + self.model_type.value + "_" + str(iter) + "_model.pt";
        torch.save(model.state_dict(), checkpoint_test_filename)

      lr_scheduler.step()

      if rospy.is_shutdown():
        break
      pass

    if rospy.is_shutdown():
      return

    # save final
    self.test_and_save_images(model, "final", is_last_test=True)
    checkpoint_test_filename = self.checkpoint_prefix + self.model_type.value + "_final_model.pt";
    torch.save(model.state_dict(), checkpoint_test_filename)
    pass
  pass # class

rospy.init_node('octree_sparse', anonymous=True)

if torch.cuda.is_available():
  rospy.loginfo("CUDA is available")
else:
  rospy.logfatal("CUDA is NOT available")
  exit(1)

torch.manual_seed(10)

#torch.autograd.set_detect_anomaly(True)

rospy.loginfo("CUDA total memory: %d MB" % (torch.cuda.get_device_properties(0).total_memory / (1024*1024)))

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
max_memory_fraction = min(float(max_memory_mb) * 1024*1024 / float(torch.cuda.get_device_properties(0).total_memory), 1.0)
rospy.loginfo("CUDA max memory fraction %f (%d MB)" % (max_memory_fraction, max_memory_mb))
torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
torch.set_default_device('cuda')

trainer = Trainer()
trainer.sparse_octree_test();
