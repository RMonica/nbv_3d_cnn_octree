#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
import pickle

import torch

import enum_config

from octree_sparse_pytorch_common import OctreeConv2D, OctreeConv3D, OctreeResblock, OctreeConv3DTORCHSPARSE
from octree_sparse_pytorch_common import LabelPredict

from octree_sparse_pytorch_utils import *

class OctreeEncoder(torch.nn.Module):
  def __init__(self, nin, channels_list, depth=6, resblock_num=3,
               is_3d=False, use_torchsparse=True, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin

    self.resblock_num = resblock_num

    self.arch_type = arch_type

    self.depth = depth
    channels = channels_list

    use_batch_norm = False

    self.is_3d = is_3d
    if self.is_3d:
      if use_torchsparse:
        self.ConvType = OctreeConv3DTORCHSPARSE
      else:
        self.ConvType = OctreeConv3D
      self.DeconvType = OctreeConv3D
    else:
      self.ConvType = OctreeConv2D
      self.DeconvType = OctreeConv2D

    self.input_convs = torch.nn.ModuleList([None] * (self.depth))
    for d in range(0, self.depth):
      input_conv = self.ConvType(3, nin=self.nin, nout=channels[d], activation=sparse_leaky_relu, batch_norm=use_batch_norm)
      self.input_convs[d] = input_conv
      pass

    self.convs = torch.nn.ModuleList([None] * (self.depth))

    if self.arch_type == enum_config.ArchType.BASE:
      self.convs2 = torch.nn.ModuleList([None] * (self.depth))

      prev_channels = channels[self.depth-1]
      for d in range(self.depth - 1, 0, -1):
        conv = self.ConvType(3, nin=prev_channels, nout=channels[d-1], stride=1, activation=sparse_leaky_relu, batch_norm=use_batch_norm)
        prev_channels = channels[d-1]
        self.convs[d] = conv
        conv2 = self.ConvType(3, nin=prev_channels, nout=prev_channels, stride=2, activation=sparse_leaky_relu, batch_norm=use_batch_norm)
        self.convs2[d] = conv2
      pass

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, (self.depth))])
      prev_channels = channels[self.depth-1]
      for d in range(self.depth - 1, 0, -1):
        for i in range(0, self.resblock_num):
          resblock = OctreeResblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d,
                                    use_torchsparse=use_torchsparse)
          self.resblocks[d].append(resblock)
        conv = self.ConvType(3, nin=prev_channels, nout=channels[d-1], stride=2, activation=sparse_leaky_relu, batch_norm=use_batch_norm)
        prev_channels = channels[d-1]
        self.convs[d] = conv
      pass

    self.nout = prev_channels
    pass

  def get_nout(self):
    return self.nout

  def forward_block(self, d, x, mask):
    if self.arch_type == enum_config.ArchType.BASE:
      x = self.convs[d](x, mask)
      mask = mask_downsample(mask, is_3d=self.is_3d)
      x = self.convs2[d](x, mask)

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x, mask)
      mask = mask_downsample(mask, is_3d = self.is_3d)
      x = self.convs[d](x, mask)
    return x, mask

  def forward(self, inputs, masks):
    x = inputs[self.depth - 1]
    mask = masks[self.depth - 1]
    skips = [None] * (self.depth)

    x = self.input_convs[self.depth - 1](x, mask)
    for d in range(self.depth - 1, 0, -1):
      skips[d] = x
      x, mask = self.forward_block(d, x, mask)

      mask = (mask + masks[d-1]).coalesce()
      x = (x + self.input_convs[d-1](inputs[d-1], masks[d-1])) / 2.0
      x = x.coalesce()
    return x, mask, skips
  pass

class OctreeDecoder(torch.nn.Module):
  def __init__(self, nin, nout, channels_list, depth=6, output_activation=sparse_sigmoid, resblock_num=3, label_hidden_channels=32,
               is_3d=False, use_torchsparse=True, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.resblock_num = resblock_num

    self.label_hidden_channels = label_hidden_channels

    self.arch_type = arch_type

    self.depth = depth
    channels = channels_list

    use_batch_norm = False

    self.deconvs = torch.nn.ModuleList([None] * (self.depth))

    self.is_3d = is_3d
    if self.is_3d:
      if use_torchsparse:
        self.ConvType = OctreeConv3DTORCHSPARSE
      else:
        self.ConvType = OctreeConv3D
      self.DeconvType = OctreeConv3D
    else:
      self.ConvType = OctreeConv2D
      self.DeconvType = OctreeConv2D

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, (self.depth))])

    prev_channels = self.nin
    for d in range(0, self.depth):
      if d >= 1:
        deconv = self.DeconvType(3, nin=prev_channels, nout=channels[d], activation=sparse_leaky_relu,
                                 batch_norm=use_batch_norm, inv_stride=2)
        prev_channels = channels[d]
        self.deconvs[d] = deconv

      if self.arch_type == enum_config.ArchType.RESBLOCK:
        for i in range(0, self.resblock_num):
          resblock = OctreeResblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d,
                                    use_torchsparse=use_torchsparse)
          self.resblocks[d].append(resblock)

    self.output_convs = torch.nn.ModuleList([None] * (self.depth))
    self.mask_predicts = torch.nn.ModuleList([None] * (self.depth))
    for d in range(0, self.depth):
      output_conv = self.ConvType(1, nin=channels[d], nout=self.nout, activation=output_activation, batch_norm=use_batch_norm)
      self.output_convs[d] = output_conv

      mask_predict = LabelPredict(nin=channels[d], nout=2, use_batch_norm=use_batch_norm,
                                  hidden_channels=self.label_hidden_channels, is_3d=self.is_3d, use_torchsparse=use_torchsparse)
      self.mask_predicts[d] = mask_predict
      pass
    pass

  def forward_block(self, d, x, mask):
    if self.arch_type == enum_config.ArchType.RESBLOCK:
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x, mask)
    return x, mask

  def forward(self, input, imask, skips, forced_masks=None, uninteresting_masks=None, training_with_unified_loss=False,
              logit_shuffle_chance=0.0):
    x = input.coalesce()
    mask = imask.coalesce()
    outputs = []
    masks = []
    logits = []

    unified_loss_data = {
      "prev_masks": [],
      "prev_outputs": [],
      "prev_logits": [],
      "this_logits": [],
      "this_masks": [],
      "this_outputs": [],
      "next_masks": [],
      "next_outputs": [],
      "next_logits": []
      }

    prev_level_logit = None;
    prev_x = None

    for d in range(0, self.depth):
      if d >= 1:
        mask = mask_upsample(mask, is_3d=self.is_3d)
        x = self.deconvs[d](x, mask)
        x = (x + eliminate_zeros_exact(skips[d] * mask)) / 2.0
        x = x.coalesce()

      x, mask = self.forward_block(d, x, mask)

      this_mask = mask
      if uninteresting_masks is not None:
        this_mask = eliminate_zero_or_less((this_mask - uninteresting_masks[d]).coalesce())

      # if last layer, predict all missing
      # otherwise, use mask_predicts
      if d+1 < self.depth:
        logit, this_mask = self.mask_predicts[d](x, this_mask, shuffle_chance=logit_shuffle_chance)
        this_mask = eliminate_zero_or_less(this_mask)
        logit = eliminate_zeros_exact(logit)

      if training_with_unified_loss and (prev_level_logit is not None):
        prev_mask = mask_downsample_max(this_mask, is_3d=self.is_3d).detach()
        prev_logit = prev_level_logit * prev_mask # prev_logit
        prev_logit = fix_dimensionality_empty_tensor(prev_logit, is_3d=self.is_3d)
        prev_output = self.output_convs[d - 1](prev_x, prev_mask)

      if training_with_unified_loss:
        if d+1 < self.depth:
          this_logit = sparse_softmax(logit, -1)
          this_logit = sparse_select(this_logit, -1, 1) # take second value
          this_logit = sparse_unsqueeze(this_logit, -1).coalesce()
          this_level_logit = this_logit
          this_logit = this_logit * this_mask
          this_logit = fix_dimensionality_empty_tensor(this_logit, is_3d=self.is_3d)
          if prev_level_logit is not None:
            this_logit = mask_upsample(prev_mask - prev_logit, is_3d=self.is_3d) * this_logit # (1 - prev_logit) * this_logit
            this_logit = fix_dimensionality_empty_tensor(this_logit, is_3d=self.is_3d)
        else:
          if prev_level_logit is not None:
            this_logit = mask_upsample(prev_mask - prev_logit, is_3d=self.is_3d) * this_mask
            this_logit = fix_dimensionality_empty_tensor(this_logit, is_3d=self.is_3d)

      if training_with_unified_loss and (d+1 < self.depth):
        # next level logit
        next_mask = mask_upsample(this_mask, is_3d=self.is_3d).detach()
        next_x = self.deconvs[d+1](x, next_mask)
        next_x = (next_x + eliminate_zeros_exact(skips[d+1] * next_mask)) / 2.0
        next_x = next_x.coalesce()
        next_x, next_mask = self.forward_block(d + 1, next_x, next_mask)
        if uninteresting_masks is not None:
          next_mask = eliminate_zero_or_less((next_mask - uninteresting_masks[d+1]).coalesce())
        next_output = self.output_convs[d + 1](next_x, next_mask)
        if prev_level_logit is not None:
          tmp_mult = mask_upsample(prev_mask - prev_logit, is_3d=self.is_3d) * (this_mask - this_logit) # (1 - prev_logit) * (1 - this_logit)
          tmp_mult = fix_dimensionality_empty_tensor(tmp_mult, is_3d=self.is_3d)
          next_logit = mask_upsample(tmp_mult, is_3d=self.is_3d)
          tmp_mult = None
        else:
          next_logit = mask_upsample(this_mask - this_logit, is_3d=self.is_3d)

      # override mask if forced
      if forced_masks is not None:
        this_mask = forced_masks[d]

      output = self.output_convs[d](x, this_mask)
      mask = mask - this_mask

      if uninteresting_masks is not None:
        mask = mask - uninteresting_masks[d]
      mask = eliminate_zero_or_less(mask.coalesce())
      masks.append(this_mask)
      outputs.append(output)
      logits.append(logit if d+1 < self.depth else None)
      if training_with_unified_loss:
#        print("-- d = %d --" % d)
#        if d >= 1:
#          print("prev_mask shape " + str(prev_mask.shape) + " count " + str(prev_mask.values().shape))
#          print("prev_output shape " + str(prev_output.shape) + " count " + str(prev_output.values().shape))
#          print("prev_logit shape " + str(prev_logit.shape) + " count " + str(prev_logit.values().shape))
#        print("this_mask shape " + str(this_mask.shape) + " count " + str(this_mask.values().shape))
#        print("output shape " + str(output.shape) + " count " + str(output.values().shape))
#        print("this_logit shape " + str(this_logit.shape) + " count " + str(this_logit.values().shape))
#        if d+1 < self.depth:
#          print("next_mask shape " + str(next_mask.shape) + " count " + str(next_mask.values().shape))
#          print("next_output shape " + str(next_output.shape) + " count " + str(next_output.values().shape))
#          print("next_logit shape " + str(next_logit.shape) + " count " + str(next_logit.values().shape))
        unified_loss_data["prev_masks"].append(prev_mask if d >= 1 else None)
        unified_loss_data["prev_outputs"].append(prev_output if d >= 1 else None)
        unified_loss_data["prev_logits"].append(prev_logit if d >= 1 else None)
        unified_loss_data["this_masks"].append(this_mask)
        unified_loss_data["this_outputs"].append(output)
        unified_loss_data["this_logits"].append(this_logit)
        unified_loss_data["next_masks"].append(next_mask if (d+1 < self.depth) else None)
        unified_loss_data["next_outputs"].append(next_output if (d+1 < self.depth) else None)
        unified_loss_data["next_logits"].append(next_logit if (d+1 < self.depth) else None)
        prev_level_logit = this_level_logit
        prev_x = x
      pass
    return outputs, masks, logits, unified_loss_data
  pass

class OctreeEncoderDecoder(torch.nn.Module):
  def __init__(self, nin, nout=1, depth=6, base_channels=16, max_channels=1000000000, resblock_num=3, label_hidden_channels=32,
               is_3d=False, use_torchsparse=True, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.is_3d = is_3d

    channels = [min(base_channels*(2**(i-1)), max_channels) for i in range(depth, 0, -1)]

    self.encoder_net = OctreeEncoder(nin=self.nin, depth=depth, is_3d=is_3d, use_torchsparse=use_torchsparse, channels_list=channels,
                                     arch_type=arch_type, resblock_num=resblock_num)
    #output_activation=sparse_tanh01  output_activation=None
    self.decoder_net = OctreeDecoder(nin=self.encoder_net.get_nout(), nout=self.nout, depth=depth,
                                     label_hidden_channels=label_hidden_channels, is_3d=is_3d, use_torchsparse=use_torchsparse,
                                     channels_list=channels, arch_type=arch_type, resblock_num=resblock_num)

  def forward(self, inputs, masks, forced_masks=None, uninteresting_masks=None, training_with_unified_loss=False):
    x, x_mask, skips = self.encoder_net(inputs, masks)
    outputs, output_masks, output_logits, unified_loss_data = self.decoder_net(x, x_mask, skips, forced_masks=forced_masks,
                                                                               uninteresting_masks=uninteresting_masks,
                                                                               training_with_unified_loss=training_with_unified_loss)
    return outputs, output_masks, output_logits, unified_loss_data
