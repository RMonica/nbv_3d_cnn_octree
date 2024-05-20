#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
import pickle

import torch

from octree_sparse_pytorch_common import pytorch_to_minkowski, minkowski_to_pytorch
import enum_config

try:
  import MinkowskiEngine as ME
  has_minkowski = True
except ImportError:
  has_minkowski = False

from octree_sparse_pytorch_utils import *

def get_minkowski_without_activation(nin, nout, kernel_size, stride, dimension, expand_coordinates=False):
  return ME.MinkowskiConvolution(nin, nout, kernel_size=kernel_size, stride=stride, dimension=dimension, bias=True,
                                 expand_coordinates=expand_coordinates).cuda()

def get_minkowski_with_relu(nin, nout, kernel_size, stride, dimension, expand_coordinates=False):
  return torch.nn.Sequential(ME.MinkowskiConvolution(nin, nout, kernel_size=kernel_size, stride=stride,
                                                     dimension=dimension, bias=True,
                                                     expand_coordinates=expand_coordinates).cuda(),
                             ME.MinkowskiReLU().cuda()
                            )

def get_minkowski_with_sigmoid(nin, nout, kernel_size, stride, dimension, expand_coordinates=False):
  return torch.nn.Sequential(ME.MinkowskiConvolution(nin, nout, kernel_size=kernel_size, stride=stride,
                                                     dimension=dimension, bias=True,
                                                     expand_coordinates=expand_coordinates).cuda(),
                             ME.MinkowskiSigmoid().cuda()
                            )

def get_minkowski_deconv_with_activation(nin, nout, kernel_size, stride, dimension):
  return torch.nn.Sequential(ME.MinkowskiGenerativeConvolutionTranspose(nin, nout, kernel_size=kernel_size, stride=stride,
                                                                        dimension=dimension, bias=True).cuda(),
                             ME.MinkowskiReLU().cuda()
                            )

class MinkowskiApplyTorchsparseMask(torch.nn.Module):
  def __init__(self, is_3d=False, check_invariants=False):
    super().__init__()
    self.is_3d = is_3d
    self.pruning = ME.MinkowskiPruning()
    self.check_invariants = check_invariants

  def forward(self, minkowski_x, mask):
    if not mask.is_coalesced():
      mask = mask.coalesce()
    # prune out of mask boundaries
    tensor_stride = torch.tensor([1, *(minkowski_x.tensor_stride)], dtype=torch.int)
    mink_coords = minkowski_x.coordinates
    indices = mink_coords // tensor_stride
    shapelist = list(mask.shape)
    if self.is_3d:
      shapelist = shapelist[0:4]
    else:
      shapelist = shapelist[0:3]
    accept = torch.logical_and(torch.all(indices < torch.tensor(shapelist, dtype=torch.int), dim=-1),
             torch.all(indices >= torch.zeros([len(shapelist), ], dtype=torch.int), dim=-1))
    if minkowski_x.coordinates.shape[0] > 0: # prevent crash in minkowski
      minkowski_x = self.pruning(minkowski_x, accept)
    # apply mask
    x = minkowski_to_pytorch(minkowski_x, mask.shape, check_invariants=self.check_invariants)
    x = eliminate_zeros_exact(fix_dimensionality_empty_tensor(x * mask, is_3d=self.is_3d))
    # workaround to force correct output shape
    restore_size_mask_value_shape = list(mask.values().shape)
    restore_size_mask_value_shape[-1] = x.shape[-1]
    restore_size_mask_shape = list(mask.shape)
    restore_size_mask_shape[-1] = x.shape[-1]
    restore_size_mask = torch.sparse_coo_tensor(mask.indices(), torch.zeros(restore_size_mask_value_shape, dtype=torch.float32),
                                                size=restore_size_mask_shape, is_coalesced=True)
    x = (x + restore_size_mask).coalesce()
    # end workaround
    result = pytorch_to_minkowski(x, stride=minkowski_x.tensor_stride[-1], coordinate_manager=minkowski_x.coordinate_manager)
    return result

class MinkowskiSafeDeconv(torch.nn.Module):
  def __init__(self, nin, nout, kernel_size=3, stride=2, is_3d=False):
    super().__init__()

    self.dims = 3 if is_3d else 2
    self.nout = nout
    self.stride = stride
    self.deconv = get_minkowski_deconv_with_activation(nin=nin, nout=nout, kernel_size=kernel_size,
                                                       dimension=self.dims, stride=stride)
    pass

  def forward(self, minkowski_x):
    if minkowski_x.coordinates.shape[0] > 0:
      next_x = self.deconv(minkowski_x)
    else:
      next_x = ME.SparseTensor(coordinates=minkowski_x.coordinates, features=torch.empty([0, self.nout]),
                               tensor_stride=(minkowski_x.tensor_stride[-1] // self.stride),
                               coordinate_manager=minkowski_x.coordinate_manager)
    return next_x
  pass

class MinkowskiSafeConv(torch.nn.Module):
  def __init__(self, nin, nout, kernel_size=3, stride=1, is_3d=False, with_relu=False, with_sigmoid=False, expand_coordinates=False):
    super().__init__()

    self.dims = 3 if is_3d else 2
    self.nout = nout
    self.stride = stride
    if with_sigmoid and with_relu:
      rospy.logfatal("MinkowskiSafeConv: cannot add both relu and sigmoid.")
      exit(1)

    if with_relu:
      self.conv = get_minkowski_with_relu(nin=nin, nout=nout, kernel_size=kernel_size,
                                          dimension=self.dims, stride=stride, expand_coordinates=expand_coordinates)
    elif with_sigmoid:
      self.conv = get_minkowski_with_sigmoid(nin=nin, nout=nout, kernel_size=kernel_size,
                                             dimension=self.dims, stride=stride, expand_coordinates=expand_coordinates)
    else:
      self.conv = get_minkowski_without_activation(nin=nin, nout=nout, kernel_size=kernel_size,
                                                   dimension=self.dims, stride=stride, expand_coordinates=expand_coordinates)
    pass

  def forward(self, minkowski_x):
    if minkowski_x.coordinates.shape[0] > 0:
      next_x = self.conv(minkowski_x)
    else:
      next_x = ME.SparseTensor(coordinates=minkowski_x.coordinates, features=torch.empty([0, self.nout]),
                               tensor_stride=(minkowski_x.tensor_stride[-1] * self.stride),
                               coordinate_manager=minkowski_x.coordinate_manager)
    return next_x
  pass

def minkowski_argmax(x):
  values = x.features
  argmax_values = torch.argmax(values, dim=-1, keepdim=True).to(dtype=torch.float32)
  return ME.SparseTensor(coordinates=x.coordinates, features=argmax_values,
                         tensor_stride=x.tensor_stride, coordinate_manager=x.coordinate_manager)

class MinkowskiResblock(torch.nn.Module):
  def __init__(self, nin, nout, bottleneck=4, use_batch_norm=False, is_3d=False):
    super().__init__()

    self.nout = nout
    self.nin = nin

    self.is_3d = is_3d

    channelb = int(max(self.nout // bottleneck, 1))

    # nin, nout, kernel_size=3, stride=1, is_3d=False, with_relu=False, with_sigmoid=False, expand_coordinates=False
    self.block1 = MinkowskiSafeConv(nin=self.nin, nout=channelb, kernel_size=1, with_relu=True, is_3d=self.is_3d)
    self.block2 = MinkowskiSafeConv(nin=channelb, nout=channelb, kernel_size=3, with_relu=True, is_3d=self.is_3d)
    self.block3 = MinkowskiSafeConv(nin=channelb, nout=self.nout, kernel_size=1, with_relu=False, is_3d=self.is_3d)

    if self.nin != self.nout:
      self.block4 = MinkowskiSafeConv(nin=self.nin, nout=self.nout, kernel_size=1, with_relu=False, is_3d=self.is_3d)
    else:
      self.block4 = None

    self.leaky_relu = ME.MinkowskiReLU().cuda()

    self.two = torch.tensor([2.0], dtype=torch.float32).cuda()
    pass

  def forward(self, input):
    x = input
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    skip_conn = input
    if self.block4:
      skip_conn = self.block4(skip_conn)
    if x.coordinates.shape[0] > 0 or skip_conn.coordinates.shape[0] > 0: # prevent crash if empty
      x = (x + skip_conn) / self.two
      x = self.leaky_relu(x)
    return x
  pass

class LabelPredict(torch.nn.Module):
  def __init__(self, nin, nout=2, hidden_channels=32, use_batch_norm=False, is_3d=False):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.is_3d = is_3d
    if self.is_3d:
      self.dims = 3
    else:
      self.dims = 2

    self.block1 = MinkowskiSafeConv(nin=self.nin, nout=hidden_channels, kernel_size=1, stride=1, is_3d=self.is_3d, with_relu=True)
    self.block2 = MinkowskiSafeConv(nin=hidden_channels, nout=self.nout, kernel_size=1, stride=1, is_3d=self.is_3d)

  def forward(self, input, mask):
    conv = self.block1(input)
    logit = self.block2(conv)
    label = minkowski_argmax(logit)
    return logit, label

  pass

class OctreeEncoder(torch.nn.Module):
  def __init__(self, nin, channels_list, depth=6, base_channels=16, resblock_num=3,
               is_3d=False, coordinate_manager=None, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin

    self.depth = depth
    channels = channels_list
    use_batch_norm = False

    self.is_3d = is_3d
    if self.is_3d:
      self.dims = 3
    else:
      self.dims = 2

    self.resblock_num = resblock_num

    self.arch_type = arch_type

    self.minkowski_apply_torchsparse_mask = MinkowskiApplyTorchsparseMask(is_3d=self.is_3d)

    self.input_convs = torch.nn.ModuleList([None] * (self.depth))
    for d in range(0, self.depth):
      input_conv = MinkowskiSafeConv(nin=self.nin, nout=channels[d], kernel_size=3, stride=1, is_3d=self.is_3d, with_relu=True)
      #input_conv = self.ConvType(3, nin=self.nin, nout=channels[d], activation=sparse_leaky_relu, batch_norm=use_batch_norm)
      self.input_convs[d] = input_conv
      pass

    self.convs = torch.nn.ModuleList([None] * (self.depth))
    self.convs2 = torch.nn.ModuleList([None] * (self.depth))

    if self.arch_type == enum_config.ArchType.BASE:
      prev_channels = channels[self.depth-1]
      for d in range(self.depth - 1, 0, -1):
        conv = MinkowskiSafeConv(nin=prev_channels, nout=channels[d-1], kernel_size=3, stride=1, is_3d=self.is_3d, with_relu=True)
        prev_channels = channels[d-1]
        self.convs[d] = conv
        conv2 = MinkowskiSafeConv(nin=prev_channels, nout=prev_channels, kernel_size=3, stride=2,
                                  is_3d=self.is_3d, expand_coordinates=True, with_relu=True)
        self.convs2[d] = conv2

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, (self.depth))])
      prev_channels = channels[self.depth-1]
      for d in range(self.depth - 1, 0, -1):
        for i in range(0, self.resblock_num):
          resblock = MinkowskiResblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d)
          self.resblocks[d].append(resblock)
        conv = MinkowskiSafeConv(nin=prev_channels, nout=channels[d-1], kernel_size=3, stride=2,
                                 is_3d=self.is_3d, expand_coordinates=True, with_relu=True)
        prev_channels = channels[d-1]
        self.convs[d] = conv
      pass

    self.nout = prev_channels

    self.two = torch.tensor([2.0], dtype=torch.float32).cuda()
    self.two_zeros = torch.zeros([2], dtype=torch.float32).cuda() # used to broadcast each one in mask into two zeros
    pass

  def get_nout(self):
    return self.nout

  def forward_block(self, d, x, mask):
    if self.arch_type == enum_config.ArchType.BASE:
      x = self.convs[d](x)
      mask = mask_downsample(mask, is_3d=self.is_3d)
      x = self.convs2[d](x)

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x)
      mask = mask_downsample(mask, is_3d = self.is_3d)
      x = self.convs[d](x)

    x = self.minkowski_apply_torchsparse_mask(x, mask)
    return x, mask

  def forward(self, inputs, masks, coordinate_manager=None):
    minkowski_inputs = [pytorch_to_minkowski(((input + masks[i] * self.two_zeros) if masks[i].values().shape[0] != 0 else input),
                                             stride=2**(self.depth-1-i),
                                             coordinate_manager=coordinate_manager)
                        for i, input in enumerate(inputs)]

    x = minkowski_inputs[self.depth - 1]
    mask = masks[self.depth - 1]
    skips = [None] * (self.depth)

    x = self.input_convs[self.depth - 1](x)
    for d in range(self.depth - 1, 0, -1):
      skips[d] = x

      x, mask = self.forward_block(d, x, mask)
      input_x = self.input_convs[d-1](minkowski_inputs[d-1])
      if input_x.coordinates.shape[0] > 0:
        x = (x + input_x)
      x = x / self.two

      mask = (mask + masks[d-1]).coalesce()
    return x, mask, skips
  pass

class OctreeDecoder(torch.nn.Module):
  def __init__(self, nin, nout, channels_list, depth=6, base_channels=16, output_activation=sparse_sigmoid,
               resblock_num=3, label_hidden_channels=32,
               is_3d=False, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.arch_type = arch_type

    self.resblock_num = resblock_num

    self.label_hidden_channels = label_hidden_channels

    self.depth = depth
    channels = channels_list
    self.channels = channels

    use_batch_norm = False

    self.deconvs = torch.nn.ModuleList([None] * (self.depth))

    self.is_3d = is_3d
    if self.is_3d:
      self.dims = 3
    else:
      self.dims = 2

    self.minkowski_apply_torchsparse_mask = MinkowskiApplyTorchsparseMask(is_3d=self.is_3d)

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, (self.depth))])

    prev_channels = self.nin
    for d in range(0, self.depth):
      if d >= 1:
        deconv = MinkowskiSafeDeconv(nin=prev_channels, nout=channels[d], kernel_size=3,
                                     is_3d=self.is_3d, stride=2)
        prev_channels = channels[d]
        self.deconvs[d] = deconv
      if self.arch_type == enum_config.ArchType.RESBLOCK:
        for i in range(0, self.resblock_num):
          resblock = MinkowskiResblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d)
          self.resblocks[d].append(resblock)

    self.output_convs = torch.nn.ModuleList([None] * (self.depth))
    self.mask_predicts = torch.nn.ModuleList([None] * (self.depth))
    for d in range(0, self.depth):
      output_conv = MinkowskiSafeConv(nin=channels[d], nout=self.nout, kernel_size=1, stride=1, is_3d=self.is_3d, with_sigmoid=True)
      self.output_convs[d] = output_conv

      mask_predict = LabelPredict(nin=channels[d], nout=2, use_batch_norm=use_batch_norm,
                                  hidden_channels=self.label_hidden_channels, is_3d=self.is_3d)
      self.mask_predicts[d] = mask_predict
      pass
    pass

  def forward_block(self, d, x, mask):
    if self.arch_type == enum_config.ArchType.RESBLOCK:
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x)
    return x, mask

  def forward(self, input, imask, skips, forced_masks=None, uninteresting_masks=None, training_with_unified_loss=False,
              logit_shuffle_chance=0.0, coordinate_manager=None):
    x = input
    mask = imask
    if not mask.is_coalesced():
      mask = mask.coalesce()
    outputs = []
    masks = []
    logits = []

    two = torch.tensor([2.0], dtype=torch.float32)

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
        x = self.deconvs[d](x)
        x = self.minkowski_apply_torchsparse_mask(x, mask)
        masked_skip = self.minkowski_apply_torchsparse_mask(skips[d], mask)
        if masked_skip.coordinates.shape[0] > 0:
          x = (x + masked_skip)
        x = x / two

      x, mask = self.forward_block(d, x, mask)

      this_mask = mask
      if uninteresting_masks is not None:
        this_mask = eliminate_zero_or_less((this_mask - uninteresting_masks[d]).coalesce())
        x = self.minkowski_apply_torchsparse_mask(x, this_mask)

      # if last layer, predict all missing
      # otherwise, use mask_predicts
      if d+1 < self.depth:
        logit, this_mask_minkowski = self.mask_predicts[d](x, this_mask)
        this_mask = eliminate_zero_or_less(minkowski_to_pytorch(this_mask_minkowski, output_shape=this_mask.shape))
        logit = minkowski_to_pytorch(logit, output_shape=this_mask.shape)

      if training_with_unified_loss and (prev_level_logit is not None):
        prev_mask = mask_downsample_max(this_mask, is_3d=self.is_3d).detach()
        prev_logit = prev_level_logit * prev_mask # prev_logit
        prev_logit = fix_dimensionality_empty_tensor(prev_logit, is_3d=self.is_3d)
        prev_x = self.minkowski_apply_torchsparse_mask(prev_x, prev_mask)
        prev_output = self.output_convs[d - 1](prev_x)
        prev_output = minkowski_to_pytorch(prev_output, prev_mask.shape)

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
        next_x = self.deconvs[d+1](x)
        next_x = self.minkowski_apply_torchsparse_mask(next_x, next_mask)
        masked_next_skip = self.minkowski_apply_torchsparse_mask(skips[d+1], next_mask)
        if masked_next_skip.coordinates.shape[0] > 0:
          next_x = (next_x + masked_next_skip)
        next_x = next_x / two
        next_x, next_mask = self.forward_block(d + 1, next_x, next_mask)
        if uninteresting_masks is not None:
          next_mask = eliminate_zero_or_less((next_mask - uninteresting_masks[d+1]).coalesce())
          next_x = self.minkowski_apply_torchsparse_mask(next_x, next_mask)
        next_output = self.minkowski_apply_torchsparse_mask(next_x, next_mask)
        next_output = self.output_convs[d + 1](next_output)
        next_output = minkowski_to_pytorch(next_output, next_mask.shape)
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

      output = self.minkowski_apply_torchsparse_mask(x, this_mask)
      output = self.output_convs[d](output)
      output = minkowski_to_pytorch(output, this_mask.shape)
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
  def __init__(self, nin, nout=1, depth=6, base_channels=16, max_channels=1000000, resblock_num=3, label_hidden_channels=32,
               is_3d=False, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.is_3d = is_3d
    if self.is_3d:
      self.dims = 3
    else:
      self.dims = 2

    channels = [min(base_channels*(2**(i-1)), max_channels) for i in range(depth, 0, -1)]

    if not has_minkowski:
      rospy.logfatal("octree_sparse_minkowski_enc_dec: Minkowski Engine not found.")
      exit(1)

    self.encoder_net = OctreeEncoder(nin=self.nin, depth=depth, is_3d=is_3d, base_channels=base_channels, channels_list=channels,
                                     arch_type=arch_type)
    self.decoder_net = OctreeDecoder(nin=self.encoder_net.get_nout(), nout=self.nout, depth=depth,
                                     label_hidden_channels=label_hidden_channels, is_3d=is_3d,
                                     base_channels=base_channels, channels_list=channels,
                                     arch_type=arch_type)

  def forward(self, inputs, masks, forced_masks=None, uninteresting_masks=None, training_with_unified_loss=False):
    coordinate_manager = ME.CoordinateManager(D=self.dims)

    x, x_mask, skips = self.encoder_net(inputs, masks, coordinate_manager=coordinate_manager)
    outputs, output_masks, output_logits, unified_loss_data = self.decoder_net(x, x_mask, skips, forced_masks=forced_masks,
                                                                               uninteresting_masks=uninteresting_masks,
                                                                               training_with_unified_loss=training_with_unified_loss,
                                                                               coordinate_manager=coordinate_manager)
    return outputs, output_masks, output_logits, unified_loss_data
