#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
import pickle

import torch

from octree_sparse_pytorch_utils import *

try:
  import torchsparse
  has_torchsparse = True
except ImportError:
  has_torchsparse = False

try:
  import MinkowskiEngine as ME
  has_minkowski = True
except ImportError:
  has_minkowski = False

class OctreeConv3DTORCHSPARSE(torch.nn.Module):
  def __init__(self, size, nin, nout, stride=1, inv_stride=1, activation=None, batch_norm=False, with_bias=True):
    super().__init__()

    self.size = size
    self.nout = nout
    self.nin = nin
    self.stride = stride
    self.activation = activation
    self.inv_stride = inv_stride
    self.with_bias = with_bias

    stdv = 1.0 / math.sqrt(float(self.size * self.size * self.nin))
    init_b = (torch.rand(size=(nout,), dtype=torch.float32) - 0.5) * 2.0 * stdv
    if self.with_bias:
      self.bias = torch.nn.parameter.Parameter(init_b)

    if not has_torchsparse:
      raise RuntimeError('OctreeConv3D: attempted to instantiate OctreeConv3D, but torchsparse module was not found.')

    if inv_stride <= stride:
      self.conv = torchsparse.nn.Conv3d(self.nin, self.nout, size, stride=stride)
    else:
      self.conv = torchsparse.nn.Conv3d(self.nin, self.nout, size, stride=inv_stride, transposed=True)

    self.batch_norm = None
    if batch_norm:
      self.batch_norm = torch.nn.BatchNorm1d(nout)
    pass

  def forward(self, input, mask):
    if input.shape[-1] != self.nin:
      raise RuntimeError('OctreeConv3D: created with %d channels, received %d channels at runtime instead.' %
                         (self.nin, input.shape[-1]))

    output_shape = (input.shape[0],
                    input.shape[1] // self.stride * self.inv_stride,
                    input.shape[2] // self.stride * self.inv_stride,
                    input.shape[3] // self.stride * self.inv_stride,
                    self.nout)

    input_indices = input.indices().to(dtype=torch.int)
    input_indices = input_indices.transpose(0, 1)
    input_values = input.values()
    torchsparse_input = torchsparse.SparseTensor(coords=input_indices, feats=input_values)

    torchsparse_output = self.conv(torchsparse_input)

    output_values = torchsparse_output.feats
    output_indices = torchsparse_output.coords.transpose(0, 1).to(dtype=torch.int64)

    result = torch.sparse_coo_tensor(output_indices, output_values, output_shape).coalesce()

    if mask.values().shape[0] != 0 and self.with_bias: # prevent crash in case of empty mask
      result = (result + mask * self.bias)
    result = result * mask
    result = fix_dimensionality_empty_tensor_3d(result)
    result = eliminate_zeros_exact(result)
    if self.batch_norm is not None:
      result = self.batch_norm(result)
    if self.activation is not None:
      result = self.activation(result)
    return result

  def __str__(self):
    return ("OctreeConv3D(%dx%d, %d channels, kernel %s, bias %s, stride %d/%d)" %
            (self.size, self.size, self.nout, str(self.kernel), str(self.bias), self.stride, self.inv_stride))
  pass

class OctreeConv3D(torch.nn.Module):
  def __init__(self, size, nin, nout, stride=1, inv_stride=1, activation=None, batch_norm=False, with_bias=True):
    super().__init__()

    self.size = size
    self.nout = nout
    self.nin = nin
    self.stride = stride
    self.activation = activation
    self.inv_stride = inv_stride
    self.with_bias = with_bias
    stdv = 1.0 / math.sqrt(float(self.size * self.size * self.size * self.nin))
    init_k = (torch.rand(size=(size, size, size, nin, nout), dtype=torch.float32) - 0.5) * 2.0 * stdv
    init_b = (torch.rand(size=(nout,), dtype=torch.float32) - 0.5) * 2.0 * stdv
    self.kernel = torch.nn.parameter.Parameter(init_k)
    if self.with_bias:
      self.bias = torch.nn.parameter.Parameter(init_b)
    self.batch_norm = None
    if batch_norm:
      self.batch_norm = torch.nn.BatchNorm1d(nout)
    pass

  def forward(self, input, mask):
    if input.shape[-1] != self.nin:
      raise RuntimeError('OctreeConv3D: created with %d channels, received %d channels at runtime instead.' %
                         (self.nin, input.shape[-1]))

    output_shape = (input.shape[0],
                    input.shape[1] // self.stride * self.inv_stride,
                    input.shape[2] // self.stride * self.inv_stride,
                    input.shape[3] // self.stride * self.inv_stride,
                    self.nout)

    result = torch.sparse_coo_tensor(torch.empty([4, 0]), torch.empty([0, self.nout]), size=output_shape, dtype=torch.float32)

    indices = input.indices()
    indices = torch.transpose(indices, 1, 0)

    zero_indices = torch.zeros(size=[4], dtype=torch.int64)
    max_indices = torch.tensor([(2**62 - 1), output_shape[1]-1, output_shape[2]-1, output_shape[3]-1], dtype=torch.int64)

    divisor = float(self.size * self.size * self.size)
    for kx in range(0, self.size):
      for ky in range(0, self.size):
        for kz in range(0, self.size):
          k = self.kernel[kz, ky, kx, :, :]
          values = torch.matmul(input.values(), k)

          shift = [0, kz - self.size // 2, ky - self.size // 2, kx - self.size // 2]
          shift_indices = indices
          if (self.inv_stride != 0) or (self.stride != 0) or (shift != [0, 0, 0]):
            shift_indices = indices.clone()
            shift_indices[:, 1:] = shift_indices[:, 1:] * self.inv_stride
            shift_indices = (shift_indices + torch.tensor(shift, dtype=torch.int64))
            shift_indices[:, 1:] = shift_indices[:, 1:] // self.stride
            shift_indices = torch.maximum(shift_indices, zero_indices)
            shift_indices = torch.minimum(shift_indices, max_indices)
          temp = torch.sparse_coo_tensor(torch.transpose(shift_indices, 1, 0), values, output_shape)
          result = (result + temp).coalesce()

          # cleanup memory
          temp = None
          values = None
          shift_indices = None
          acceptable_indices = None
          values = None
          pass
        pass
      pass

    if result.shape[0:-1] != mask.shape[0:-1]:
      raise RuntimeError('OctreeConv3D: output has shape %s, but mask has shape %s.' %
                         (str(tuple(result.shape[0:-1])), str(tuple(mask.shape[0:-1]))))

    #result = result / divisor
    if mask.values().shape[0] != 0 and self.with_bias: # prevent crash in case of empty mask
      result = (result + mask * self.bias)
    result = result * mask
    result = fix_dimensionality_empty_tensor_3d(result)
    result = eliminate_zeros_exact(result)
    if self.batch_norm is not None:
      result = self.batch_norm(result)
    if self.activation is not None:
      result = self.activation(result)
    return result

  def __str__(self):
    return ("OctreeConv3D(%dx%d, %d channels, kernel %s, bias %s, stride %d/%d)" %
            (self.size, self.size, self.nout, str(self.kernel), str(self.bias), self.stride, self.inv_stride))
  pass

class OctreeConv2D(torch.nn.Module):
  def __init__(self, size, nin, nout, stride=1, inv_stride=1, activation=None, batch_norm=False, with_bias=True):
    super().__init__()

    self.size = size
    self.nout = nout
    self.nin = nin
    self.stride = stride
    self.activation = activation
    self.inv_stride = inv_stride
    self.with_bias = with_bias
    stdv = 1.0 / math.sqrt(float(self.size * self.size * self.nin))
    init_k = (torch.rand(size=(size, size, nin, nout), dtype=torch.float32) - 0.5) * 2.0 * stdv
    init_b = (torch.rand(size=(nout,), dtype=torch.float32) - 0.5) * 2.0 * stdv
    self.kernel = torch.nn.parameter.Parameter(init_k)
    if self.with_bias:
      self.bias = torch.nn.parameter.Parameter(init_b)
    self.batch_norm = None
    if batch_norm:
      self.batch_norm = torch.nn.BatchNorm1d(nout)
    pass

  def forward(self, input, mask):
    if input.shape[-1] != self.nin:
      raise RuntimeError('OctreeConv2D: created with %d channels, received %d channels at runtime instead.' %
                         (self.nin, input.shape[-1]))

    output_shape = (input.shape[0],
                    input.shape[1] // self.stride * self.inv_stride,
                    input.shape[2] // self.stride * self.inv_stride,
                    self.nout)

    result = torch.sparse_coo_tensor(torch.empty([3, 0]), torch.empty([0, self.nout]), size=output_shape, dtype=torch.float32)

    indices = input.indices()
    indices = torch.transpose(indices, 1, 0)

    zero_indices = torch.zeros(size=[3], dtype=torch.int64)
    max_indices = torch.tensor([(2**62 - 1), output_shape[1]-1, output_shape[2]-1], dtype=torch.int64)

    divisor = float(self.size * self.size)
    for kx in range(0, self.size):
      for ky in range(0, self.size):
        k = self.kernel[ky, kx, :, :]
        values = torch.matmul(input.values(), k)

        shift = [0, ky - self.size // 2, kx - self.size // 2]
        shift_indices = indices
        if (self.inv_stride != 0) or (self.stride != 0) or (shift != [0, 0, 0]):
          shift_indices = indices.clone()
          shift_indices[:, 1:] = shift_indices[:, 1:] * self.inv_stride
          shift_indices = (shift_indices + torch.tensor(shift, dtype=torch.int64))
          shift_indices[:, 1:] = shift_indices[:, 1:] // self.stride
          shift_indices = torch.maximum(shift_indices, zero_indices)
          shift_indices = torch.minimum(shift_indices, max_indices)
#          acceptable_indices = torch.logical_and(shift_indices >= torch.tensor([0, 0], dtype=torch.int64),
#                                                 shift_indices < torch.tensor([output_shape[0], output_shape[1]]))
#          acceptable_indices = torch.all(acceptable_indices, 1)
#          shift_indices = torch.masked_select(shift_indices, acceptable_indices.unsqueeze(1).expand(-1, 2)).reshape((-1, 2))
#          values = torch.masked_select(values, acceptable_indices.unsqueeze(1).expand(-1, self.nout)).reshape((-1, self.nout))
        temp = torch.sparse_coo_tensor(torch.transpose(shift_indices, 1, 0), values, output_shape)
        result = (result + temp).coalesce()

        # cleanup memory
        temp = None
        values = None
        shift_indices = None
        acceptable_indices = None
        values = None
        pass
      pass

    if result.shape[0:-1] != mask.shape[0:-1]:
      raise RuntimeError('OctreeConv2D: output has shape %s, but mask has shape %s.' %
                         (str(tuple(result.shape[0:-1])), str(tuple(mask.shape[0:-1]))))

    #result = result / divisor
    if mask.values().shape[0] != 0 and self.with_bias: # prevent crash in case of empty mask
      result = (result + mask * self.bias)
    result = result * mask
    result = fix_dimensionality_empty_tensor_2d(result)
    result = eliminate_zeros_exact(result)
    if self.batch_norm is not None:
      result = self.batch_norm(result)
    if self.activation is not None:
      result = self.activation(result)
    return result

  def __str__(self):
    return ("OctreeConv2D(%dx%d, %d channels, kernel %s, bias %s, stride %d/%d)" %
            (self.size, self.size, self.nout, str(self.kernel), str(self.bias), self.stride, self.inv_stride))
  pass

def pytorch_to_minkowski(x, stride=1, coordinate_manager=None):
  if not x.is_coalesced():
    x = x.coalesce()
  input_indices = x.indices().clone()
  if input_indices.shape[1] != 0: # prevent crash if empty
    input_indices[1:] = input_indices[1:] * torch.tensor([stride], dtype=torch.int64)
  input_indices = input_indices.transpose(0, 1).to(dtype=torch.int).contiguous()
  input_values = x.values()
  result = ME.SparseTensor(coordinates=input_indices, features=input_values,
                           tensor_stride=stride, coordinate_manager=coordinate_manager)
  return result

def minkowski_to_pytorch(x, output_shape, check_invariants=True):
  stride = x.tensor_stride[0]
  output_values = x.features
  output_indices = x.coordinates.transpose(0, 1).to(dtype=torch.int64)
  output_indices[1:] = output_indices[1:] // stride
  output_shape = list(output_shape)
  output_shape[-1] = output_values.shape[-1]
  result = torch.sparse_coo_tensor(output_indices, output_values, output_shape, check_invariants=check_invariants).coalesce()
  return result

class Resblock(torch.nn.Module):
  def __init__(self, nin, nout, bottleneck = 4, use_batch_norm=False, is_3d=False):
    super().__init__()

    self.nout = nout
    self.nin = nin

    self.is_3d = is_3d

    if self.is_3d:
      self.convtype = torch.nn.Conv3d
    else:
      self.convtype = torch.nn.Conv2d

    channelb = int(max(self.nout // bottleneck, 1))

    self.block1 = self.convtype(self.nin, channelb, 1, padding='same', dtype=torch.float32)
    self.block2 = self.convtype(channelb, channelb, 3, padding='same', dtype=torch.float32)
    self.block3 = self.convtype(channelb, self.nout, 1, padding='same', dtype=torch.float32)

    self.block4 = None
    if self.nin != self.nout:
      self.block4 = self.convtype(self.nin, self.nout, 1, padding='same', dtype=torch.float32)
    pass

  def forward(self, input):
    x = input
    x = self.block1(x)
    x = torch.nn.functional.leaky_relu(x)
    x = self.block2(x)
    x = torch.nn.functional.leaky_relu(x)
    x = self.block3(x)
    skip_conn = input
    if self.block4:
      skip_conn = self.block4(skip_conn)
    x = (x + skip_conn) / 2.0
    x = torch.nn.functional.leaky_relu(x)
    return x
  pass

class OctreeResblock(torch.nn.Module):
  def __init__(self, nin, nout, bottleneck=4, use_batch_norm=False, is_3d=False, use_torchsparse=True):
    super().__init__()

    self.nout = nout
    self.nin = nin

    self.is_3d = is_3d
    if self.is_3d:
      if use_torchsparse:
        self.ConvType = OctreeConv3DTORCHSPARSE
      else:
        self.ConvType = OctreeConv3D
    else:
      self.ConvType = OctreeConv2D

    channelb = int(max(self.nout // bottleneck, 1))

    self.block1 = self.ConvType(1, nin=self.nin, nout=channelb, activation=sparse_leaky_relu, batch_norm=use_batch_norm)
    self.block2 = self.ConvType(3, nin=channelb, nout=channelb, activation=sparse_leaky_relu, batch_norm=use_batch_norm)
    self.block3 = self.ConvType(1, nin=channelb, nout=self.nout, activation=None, batch_norm=use_batch_norm)

    if self.nin != self.nout:
      self.block4 = self.ConvType(1, nin=self.nin, nout=self.nout, activation=None, batch_norm=use_batch_norm)
    else:
      self.block4 = None
    pass

  def forward(self, input, mask):
    x = input
    x = self.block1(x, mask)
    x = self.block2(x, mask)
    x = self.block3(x, mask)
    skip_conn = input
    if self.block4:
      skip_conn = self.block4(skip_conn, mask)
    x = (x + skip_conn) / 2.0
    x = x.coalesce()
    x = sparse_leaky_relu(x)
    return x
  pass

def label_shuffle_logit(logit, chance, coalesce = True):
  if not logit.is_coalesced:
    logit = logit.coalesce()
  values = logit.values()
  indices = logit.indices()

  mask_to_change1 = (torch.rand(size=(values.shape[0], )) < chance)
  mask_to_change2 = (torch.rand(size=(values.shape[0], )) < chance)
  mask_to_change2 = mask_to_change2.logical_and(mask_to_change1.logical_not())
  mask_to_change = mask_to_change1.logical_or(mask_to_change2)
  multiplier = torch.ones(values.shape, dtype=torch.float32)
  multiplier[mask_to_change] = torch.tensor([0, 0], dtype=torch.float32)
  addend1 = torch.zeros(values.shape, dtype=torch.float32)
  addend1[mask_to_change1] = torch.tensor([1, 0], dtype=torch.float32)
  addend2 = torch.zeros(values.shape, dtype=torch.float32)
  addend2[mask_to_change2] = torch.tensor([0, 1], dtype=torch.float32)

  values = values * multiplier + addend1 + addend2
  result = torch.sparse_coo_tensor(indices, values, size=logit.shape)
  if coalesce:
    result = result.coalesce()
  return result

class LabelPredict(torch.nn.Module):
  def __init__(self, nin, nout=2, hidden_channels=32, use_batch_norm=False, is_3d=False, use_torchsparse=True):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.is_3d = is_3d
    if self.is_3d:
      if use_torchsparse:
        self.ConvType = OctreeConv3DTORCHSPARSE
      else:
        self.ConvType = OctreeConv3D
    else:
      self.ConvType = OctreeConv2D

    self.block1 = self.ConvType(1, nin=self.nin, nout=hidden_channels, activation=sparse_leaky_relu,
                                batch_norm=use_batch_norm)
    self.block2 = self.ConvType(1, nin=hidden_channels, nout=self.nout, activation=None,
                                batch_norm=use_batch_norm)

  def forward(self, input, mask, shuffle_chance=0.0):
    conv = self.block1(input, mask)
    logit = self.block2(conv, mask)
    if shuffle_chance != 0.0:
      logit = label_shuffle_logit(logit, shuffle_chance)
    label = sparse_argmax(logit, keepdim=True).to(dtype=torch.float32)
    return logit, label

  pass

class SparseLogitsLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    pass

  def forward(self, logits, gt_logits):
    loss = 0.0
    for logit, gt_logit in zip(logits, gt_logits):
      if logit is None:
        continue
      if (logit.values().shape[0] != gt_logit.values().shape[0]) or not bool(torch.all(logit.indices() == gt_logit.indices())):
        raise RuntimeError("SparseLogitsLoss: logits have different indices, something is wrong.\n" +
                           "logit:\n" + str(logit) + "\ngt_logit:\n" + str(gt_logit))
      logit_values = logit.values()
      gt_logit_values = gt_logit.values()
      loss += self.cross_entropy_loss(logit_values, gt_logit_values)
    return loss

class SparseOutputLoss(torch.nn.Module):
  def __init__(self, dims=2):
    super().__init__()

    self.dims = dims
    self.is_3d = (dims == 3)

    self.criterion = torch.nn.MSELoss(reduction='mean')
    pass

  def forward(self, outputs, output_masks, image_pyramids, gts, gt_masks):
    #multiplier = 2.0**(len(outputs)-1)
    total_multiplier = 0.0
    total_loss = torch.zeros(size=[1])

    image_pyramid = image_pyramids[0]
    sq_image_pyramid = image_pyramids[1]
    weight_pyramid = image_pyramids[2]

    if ((len(outputs) != len(output_masks)) or
        (len(output_masks) != len(image_pyramid)) or
        (len(image_pyramid) != len(sq_image_pyramid))):
      raise RuntimeError('SparseOutputLoss: mismatched input lengths %d, %d, %d, %d' %
                         (len(outputs), len(output_masks), len(image_pyramid), len(sq_image_pyramid)))

    for o, om, gt_img, sq_gt_img, w_img, gt, gtm in zip(outputs, output_masks, image_pyramid, sq_image_pyramid, weight_pyramid, gts, gt_masks):
      sel_gt = gt_img * om
      sel_gt = eliminate_zeros_exact(sel_gt)
      sel_gt = fix_dimensionality_empty_tensor(sel_gt, is_3d=self.is_3d)
      sel_sq_gt = sq_gt_img * om
      sel_sq_gt = eliminate_zeros_exact(sel_sq_gt)
      sel_sq_gt = fix_dimensionality_empty_tensor(sel_sq_gt, is_3d=self.is_3d)

      sq_o = torch.square(o)

      #diff = ((o * om) - (gt * gtm)).coalesce()
      #l = torch.sum(torch.pow(diff.values(), 2.0))
      #print("old l: " + str(l))
      l = torch.sum((sel_sq_gt + sq_o - 2.0 * sel_gt * o) * w_img)
      total_loss += l
      total_multiplier += torch.mean(w_img.values())
      #multiplier /= 2.0
      pass
    return torch.sqrt(total_loss / total_multiplier)

class SparseUnifiedOutputLoss(torch.nn.Module):
  def __init__(self, dims=2, unified_alpha=0.9, mse_leak=0.0, track_partial_losses=True):
    super().__init__()

    self.dims = dims
    self.is_3d = (dims == 3)

    self.alpha = unified_alpha # loss multiplier if decreasing a level
    self.mse_leak = mse_leak # if RMSE is zero, add a small value anyway to "punish" small voxels

    self.criterion = torch.nn.MSELoss(reduction='mean')

    self.track_partial_losses = track_partial_losses
    self.last_partial_losses = None
    pass

  def forward(self, unified_loss_data, image_pyramids, gts, gt_masks, logit_leak=0.5):
    total_multiplier = 0.0
    total_loss = torch.zeros(size=[1])

    prev_outputs = unified_loss_data["prev_outputs"]
    prev_masks = unified_loss_data["prev_masks"]
    prev_logits = unified_loss_data["prev_logits"]

    this_outputs = unified_loss_data["this_outputs"]
    this_masks = unified_loss_data["this_masks"]
    this_logits = unified_loss_data["this_logits"]

    next_outputs = unified_loss_data["next_outputs"]
    next_masks = unified_loss_data["next_masks"]
    next_logits = unified_loss_data["next_logits"]

    multiplier = 2.0**(len(this_outputs)-1)

    image_pyramid = image_pyramids[0]
    sq_image_pyramid = image_pyramids[1]
    weight_pyramid = image_pyramids[2]

    if ((len(this_outputs) != len(image_pyramid)) or
        (len(image_pyramid) != len(sq_image_pyramid))):
      raise RuntimeError('SparseUnifiedOutputLoss: mismatched input lengths %d, %d, %d' %
                         (len(unified_loss_data["this_outputs"]), len(image_pyramid), len(sq_image_pyramid)))

    total_partial_losses = [0.0] * 3
    for d, (gt, gtm) in enumerate(zip(gts, gt_masks)):

      l = torch.zeros(size=[1])

      best_debug_loss = 0.0
      best_debug_loss_index = -1

      num_not_none = 0.0
      for dd in [0, 1, 2]: # prev, this, next
        o = ([prev_outputs[d], this_outputs[d], next_outputs[d]])[dd]
        if not (o is None):
          num_not_none += 1.0
      if num_not_none < 2.0:
        print("SparseUnifiedOutputLoss: unexpected num_not_none = " + str(num_not_none))
        exit(1)

      partial_losses = [0.0] * 3
      for dd in [0, 1, 2]: # prev, this, next
        #print("-- depth %d, diff depth %d --" % (d, dd))
        om = ([prev_masks[d], this_masks[d], next_masks[d]])[dd]
        o = ([prev_outputs[d], this_outputs[d], next_outputs[d]])[dd]
        ol = ([prev_logits[d], this_logits[d], next_logits[d]])[dd]

        if o is None:
          #print("Found None output at depth %d, diff depth %d" % (d, dd))
          continue

        #local_multiplier = (([(2.0 ** self.dims) * self.alpha, 1.0, (0.5 ** self.dims) / self.alpha])[dd])
        local_multiplier = (([self.alpha, 1.0, 1.0 / self.alpha])[dd])

        gt_img = image_pyramid[d + dd - 1]
        sq_gt_img = sq_image_pyramid[d + dd - 1]
        w_img = weight_pyramid[d + dd - 1]

        sel_gt = gt_img * om
        sel_gt = eliminate_zeros_exact(sel_gt)
        sel_gt = fix_dimensionality_empty_tensor(sel_gt, is_3d=self.is_3d)
        sel_sq_gt = sq_gt_img * om
        sel_sq_gt = eliminate_zeros_exact(sel_sq_gt)
        sel_sq_gt = fix_dimensionality_empty_tensor(sel_sq_gt, is_3d=self.is_3d)

        mse_leak = self.mse_leak
        ol = torch.sparse_coo_tensor(ol.indices(), ol.values() * (1.0 - logit_leak) +
                                                   (logit_leak / num_not_none), size=ol.shape).coalesce()

        mask_count = om.values().shape[0]

        ll = torch.sum(((sel_sq_gt + torch.square(o) - 2.0 * sel_gt * o) * (1.0 - mse_leak) * ol +
                          ol * mse_leak) * w_img) * local_multiplier
        #ll_wo_ol = torch.sum((sel_sq_gt + torch.square(o) - 2.0 * sel_gt * o) * w_img) * local_multiplier
#        print("d=%d dd=%d ll=%f ll_wo_ol=%f mult=%f mean_ol=%f mean_o=%f mean_gt=%f mean_square_gt=%f" %
#              (d, dd, float(ll), float(ll_wo_ol), local_multiplier, float(torch.mean(ol.values())), float(torch.mean(o.values())),
#              float(torch.mean(sel_gt.values())), float(torch.mean(sel_sq_gt.values()))))
        l += ll
        if self.track_partial_losses:
          detached_o = o.detach()
          partial_losses[dd] += float(torch.sum((sel_sq_gt + torch.square(detached_o) - 2.0 * sel_gt * detached_o) *
                                      w_img) * local_multiplier)
        #print("lm %f, mc: %d sl: %f ll: %f dl: %f " % (local_multiplier, mask_count, float(sum_ol), float(ll), float(debug_loss)))
      #print("mult %f l %f bdl: %d" % ((multiplier ** self.dims), l * (multiplier ** self.dims), best_debug_loss_index))
      total_loss += l * ((1.0 / self.alpha) ** d)
      total_multiplier += (multiplier ** self.dims)
      multiplier /= 2.0
      if self.track_partial_losses:
        for dd in range(0, 3):
          total_partial_losses[dd] += partial_losses[dd]
      pass
    if self.track_partial_losses:
      self.last_partial_losses = total_partial_losses
    return total_loss / total_multiplier

  def get_last_partial_losses(self):
    return self.last_partial_losses
