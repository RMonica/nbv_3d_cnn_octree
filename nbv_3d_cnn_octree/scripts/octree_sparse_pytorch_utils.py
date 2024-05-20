#!/usr/bin/python3
# This Python file uses the following encoding: utf-8

import os
import sys
import datetime
import time

import numpy as np
import math
import struct

import rospy

import torch

def fix_dimensionality_empty_tensor_2d(x):
  if not x.is_coalesced():
    x = x.coalesce()
  if list(x.values().shape) == [0] and list(x.indices().shape) == [4, 0]:
    x = torch.sparse_coo_tensor(torch.empty([3, 0], dtype=torch.int64), torch.empty([0, 1]), size=x.shape).coalesce()
  return x

def fix_dimensionality_empty_tensor_3d(x):
  if not x.is_coalesced():
    x = x.coalesce()
  if list(x.values().shape) == [0] and list(x.indices().shape) == [5, 0]:
    x = torch.sparse_coo_tensor(torch.empty([4, 0], dtype=torch.int64), torch.empty([0, 1]), size=x.shape).coalesce()
  return x

def fix_dimensionality_empty_tensor(x, is_3d):
  if is_3d:
    return fix_dimensionality_empty_tensor_3d(x)
  else:
    return fix_dimensionality_empty_tensor_2d(x)

def print_max_sparse_v(x, name):
  if (x.values().shape[0] == 0):
    print("%s is empty" % name)
  else:
    print("max %s: %s (%d elements)" % (name,
          str(torch.max(x.coalesce().values()).cpu().detach().numpy().squeeze()), int(x.values().shape[0])))
  pass

def mask_upsample(input_mask, multiplier=2, is_3d=False):
  if not input_mask.is_coalesced():
    input_mask = input_mask.coalesce() # make sure coalesced

  output_shape = list(input_mask.size())
  output_shape[-2] *= multiplier
  output_shape[-3] *= multiplier
  if is_3d:
    output_shape[-4] *= multiplier

  result_values = torch.empty([0, output_shape[-1]])
  result_indices = torch.empty([0, (len(output_shape)-1)])

  input_indices = input_mask.indices()
  input_indices = torch.transpose(input_indices, 1, 0)

  values = input_mask.values()
  result = None

  for kz in range(0, multiplier if is_3d else 1):
    for kx in range(0, multiplier):
      for ky in range(0, multiplier):
        new_indices = input_indices.clone()
        new_indices[:, 1:] = new_indices[:, 1:] * multiplier
        if not is_3d:
          new_indices = new_indices + torch.tensor([0, ky, kx], dtype=torch.int64)
        else:
          new_indices = new_indices + torch.tensor([0, kz, ky, kx], dtype=torch.int64)
        new_indices = torch.transpose(new_indices, 1, 0)
        if result is None:
          result = torch.sparse_coo_tensor(new_indices, values, size=output_shape, dtype=torch.float32)
        else:
          result = result + torch.sparse_coo_tensor(new_indices, values, size=output_shape, dtype=torch.float32)
          result = result.coalesce()
        #indices = (input_indices * multiplier + torch.tensor([0, ky, kx], dtype=torch.int64))
        #result_values = torch.cat([result_values, values], dim=0)
        #result_indices = torch.cat([result_indices, indices], dim=0)

  return result

# nearest neighbor downsampling
def mask_downsample(input_mask, divisor=2, is_3d=False):
  output_shape = input_mask.size()
  if not is_3d:
    output_shape = (output_shape[0], output_shape[1] // divisor, output_shape[2] // divisor, output_shape[3])
  else:
    output_shape = (output_shape[0], output_shape[1] // divisor, output_shape[2] // divisor,
                    output_shape[3] // divisor, output_shape[4])

  input_indices = input_mask.indices()
  input_indices = torch.transpose(input_indices, 1, 0)
  input_values = input_mask.values()
  acceptable_indices = torch.all(torch.remainder(input_indices[:, 1:], divisor) == 0, dim=1)

  indices = input_indices[acceptable_indices]
  indices[:, 1:] = indices[:, 1:] // divisor
  values = input_values[acceptable_indices]
  result = torch.sparse_coo_tensor(torch.transpose(indices, 1, 0), values, size=output_shape, dtype=torch.float32).coalesce()
  return result

# max MASK downsampling
def mask_downsample_max(input_mask, divisor=2, is_3d=False):
  output_shape = input_mask.size()
  if not is_3d:
    output_shape = (output_shape[0], output_shape[1] // divisor, output_shape[2] // divisor, output_shape[3])
  else:
    output_shape = (output_shape[0], output_shape[1] // divisor, output_shape[2] // divisor,
                    output_shape[3] // divisor, output_shape[4])

  indices = input_mask.indices().clone()
  indices[1:] = indices[1:] // 2

  values = input_mask.values()

  result = torch.sparse_coo_tensor(indices, values, size=output_shape, dtype=torch.float32).coalesce()
  values = result.values().clamp(0.0, 1.0) # ensure no value above 1
  indices = result.indices()
  result = torch.sparse_coo_tensor(indices, values, size=output_shape, dtype=torch.float32).coalesce()
  return result

def sparse_leaky_relu(sparse_input, coalesce=True):
  values = sparse_input.values()
  values = torch.nn.functional.leaky_relu(values)
  result = torch.sparse_coo_tensor(sparse_input.indices(), values, size=sparse_input.shape)
  if coalesce:
    result = result.coalesce()
  return result

def sparse_tanh01(sparse_input, coalesce=True):
  values = sparse_input.values()
  values = ((torch.tanh(values) + 1.0) / 2.0)
  result = torch.sparse_coo_tensor(sparse_input.indices(), values, size=sparse_input.shape)
  if coalesce:
    result = result.coalesce()
  return result

def sparse_sigmoid(sparse_input, coalesce=True):
  values = sparse_input.values()
  values = torch.nn.functional.sigmoid(values)
  result = torch.sparse_coo_tensor(sparse_input.indices(), values, size=sparse_input.shape)
  if coalesce:
    result = result.coalesce()
  return result

def sparse_argmax(sparse_input, dense_dim=0, coalesce=True, keepdim=False):
  values = sparse_input.values()
  values = torch.argmax(values, dim=dense_dim+1, keepdim=keepdim)
  if keepdim:
    output_shape = (*list(sparse_input.shape[0:-1]), 1)
  else:
    output_shape = (*list(sparse_input.shape[0:-1]), )
  result = torch.sparse_coo_tensor(sparse_input.indices(), values, size=output_shape)
  if coalesce:
    result = result.coalesce()
  return result

def sparse_softmax(x, dim=-1, coalesce=True):
  if dim < 0:
    dim += len(x.shape)

  values = x.values()
  indices = x.indices()
  sparse_dims = indices.shape[0]
  if dim < sparse_dims:
    print("NIY")
    exit(1)

  dim = dim - sparse_dims + 1
  values = torch.nn.functional.softmax(values, dim=dim)
  result = torch.sparse_coo_tensor(indices, values, size=x.shape, is_coalesced=True)
  return result

def eliminate_zeros(x, coalesce=True):
  mask = (torch.abs(x.values()) > float(0.01))
  mask = mask.squeeze(dim=1)
  values = x.values()[mask, :]
  indices = x.indices()[:, mask]
  result = torch.sparse_coo_tensor(indices, values, size=x.shape)
  if coalesce:
    result = result.coalesce()
  return result

def eliminate_zeros_exact(x, coalesce=True):
  if not x.is_coalesced():
    x = x.coalesce()
  channels = x.shape[-1]
  mask = (x.values() != torch.zeros([channels,], dtype=torch.float32))
  mask = mask.any(dim=-1, keepdim=False)
  values = x.values()[mask, :]
  indices = x.indices()[:, mask]
  result = torch.sparse_coo_tensor(indices, values, size=x.shape)
  if coalesce:
    result = result.coalesce()
  return result

def eliminate_zero_or_less(x, coalesce=True):
  if not x.is_coalesced():
    x = x.coalesce()

  mask = (x.values() > float(0.01))
  mask = mask.squeeze(dim=1)
  values = x.values()[mask, :]
  indices = x.indices()[:, mask]
  result = torch.sparse_coo_tensor(indices, values, size=x.shape)
  if coalesce:
    result = result.coalesce()
  return result

def sparse_unsqueeze(x, dim=-1, coalesce=True):
  if dim < 0:
    dim += len(x.shape) + 1

  if not x.is_coalesced():
    x = x.coalesce()

  out_shape = list(x.shape)
  out_shape.insert(dim, 1)

  values = x.values()
  indices = x.indices()
  sparse_dims = indices.shape[0]
  if dim < sparse_dims:
    result = x.unsqueeze(dim) # unsqueeze works well on indices
  else: # we need to unsqueeze values
    values = values.unsqueeze(dim - sparse_dims + 1)
    result = torch.sparse_coo_tensor(indices, values, size=out_shape)
    if coalesce:
      result = result.coalesce()
  return result

def sparse_select(x, dim=-1, index=0, coalesce=True, remove_dim=True):
  if dim < 0:
    dim += len(x.shape)

  if not x.is_coalesced():
    x = x.coalesce()

  values = x.values()
  indices = x.indices()
  sparse_dims = indices.shape[0]

  new_shape = list(x.shape)
  if remove_dim:
    new_shape.pop(dim)
  else:
    new_shape[dim] = 1

  if dim < sparse_dims:
    # TODO: untested
    mask = (indices[dim] == index)
    values = values[mask, :]
    indices = indices[:, mask]
    indices = torch.cat([indices[:dim], indices[dim+1:]], dim=1)
    result = torch.sparse_coo_tensor(indices, values, size=new_shape)
  else:
    nv = values.select(dim - sparse_dims + 1, index)
    if not remove_dim:
      nv = nv.unsqueeze(dim - sparse_dims + 1)
    result = torch.sparse_coo_tensor(indices, nv, size=new_shape, is_coalesced=True)

  return result

def sparse_clamp(sparse_tensor, min, max):
  if not sparse_tensor.is_coalesced():
    sparse_tensor = sparse_tensor.coalesce()
  values = sparse_tensor.values().clamp(min, max)
  indices = sparse_tensor.indices()
  return torch.sparse_coo_tensor(indices, values, size=sparse_tensor.shape, dtype=sparse_tensor.dtype).coalesce()

def scale_values_octree(imgs, masks, scale=1.0):
  imgs = [torch.sparse_coo_tensor(img.indices(), img.values() * scale, size=img.shape,
                                  dtype=img.dtype).coalesce() for img in imgs]
  return imgs, masks

# operations:
# "add_and_clamp_01": adds the octrees and clamps the results in [0, 1], or [clamp_min, clamp_max]
# "subtract_and_clamp_01": adds the octrees and clamps the results in [0, 1], or [clamp_min, clamp_max]
# "multiply": multiplies
# "max": maximum
def merge_octrees(imgs1, masks1, imgs2, masks2, operation="add_and_clamp_01", is_3d=False, clamp_min=0.0, clamp_max=1.0):
  num_levels = len(imgs1)
  if num_levels != len(imgs1) or num_levels != len(imgs2) or num_levels != len(masks1) or num_levels != len(masks2):
    rospy.logfatal("octree_save_load: merge_octrees: mismatching num_levels [%d, %d, %d, %d]" %
                   (len(imgs1), len(imgs2), len(masks1), len(masks2)))
    exit(1)

  imgs = [None,] * num_levels
  masks = [None,] * num_levels
  img1_to_next_level = None
  img2_to_next_level = None
  mask1_to_next_level = None
  mask2_to_next_level = None
  for li in range(0, num_levels):
    mask1 = masks1[li]
    mask2 = masks2[li]
    img1 = imgs1[li]
    img2 = imgs2[li]
    if li != 0:
      mask1 = mask1 + mask1_to_next_level
      mask2 = mask2 + mask2_to_next_level
      img1 = img1 + img1_to_next_level
      img2 = img2 + img2_to_next_level

    if li != (num_levels-1):
      mask_in_1_not_in_2 = eliminate_zero_or_less(mask1 - mask2)
      mask_in_2_not_in_1 = eliminate_zero_or_less(mask2 - mask1)
      mask_common = eliminate_zeros_exact(fix_dimensionality_empty_tensor(mask2 * mask1, is_3d=is_3d))
      values_in_1_not_in_2 = eliminate_zeros_exact(fix_dimensionality_empty_tensor(img1 * mask_in_1_not_in_2, is_3d=is_3d))
      values_in_2_not_in_1 = eliminate_zeros_exact(fix_dimensionality_empty_tensor(img2 * mask_in_2_not_in_1, is_3d=is_3d))
      img1_to_next_level = mask_upsample(values_in_1_not_in_2, is_3d=is_3d)
      img2_to_next_level = mask_upsample(values_in_2_not_in_1, is_3d=is_3d)
      mask1_to_next_level = mask_upsample(mask_in_1_not_in_2, is_3d=is_3d)
      mask2_to_next_level = mask_upsample(mask_in_2_not_in_1, is_3d=is_3d)
    else:
      mask_common = sparse_clamp(mask1 + mask2, 0.0, 1.0) # if last layer, output all

    if operation == "add_and_clamp_01":
      imgs[li] = eliminate_zeros_exact(fix_dimensionality_empty_tensor((img1 + img2) * mask_common, is_3d=is_3d))
      imgs[li] = sparse_clamp(imgs[li], clamp_min, clamp_max)
      masks[li] = mask_common
    elif operation == "subtract_and_clamp_01":
      imgs[li] = eliminate_zeros_exact(fix_dimensionality_empty_tensor((img1 - img2) * mask_common, is_3d=is_3d))
      imgs[li] = sparse_clamp(imgs[li], clamp_min, clamp_max)
      masks[li] = mask_common
    elif operation == "multiply":
      imgs[li] = eliminate_zeros_exact(fix_dimensionality_empty_tensor((img1 * img2) * mask_common, is_3d=is_3d))
      masks[li] = mask_common
    else:
      rospy.logfatal("octree_save_load: merge_octrees: unknown operation: %s" % operation)
      exit(1)
    pass

  return imgs, masks

def find_final_mask(masks, is_3d):
  depth = len(masks)
  mask = masks[depth - 1]

  for d in range(depth, 1, -1):
    mask = mask_downsample(mask, is_3d=is_3d)
    mask = (mask + masks[d-2]).coalesce()
  return mask

def logits_from_masks(forced_masks, initial_mask=None, uninteresting_masks=None, is_3d=False):
  if initial_mask is None:
    mask = torch.ones(size=masks[0].shape, dtype=torch.float32).to_sparse(sparse_dim=2)
  else:
    mask = initial_mask

  depth = len(forced_masks)
  logits = []

  for d in range(0, depth):
    if d > 0:
      mask = mask_upsample(mask, is_3d=is_3d)

    this_mask = mask
    if uninteresting_masks is not None:
      this_mask = eliminate_zero_or_less((this_mask - uninteresting_masks[d]).coalesce())

    true_mask = forced_masks[d] * this_mask
    false_mask = eliminate_zero_or_less((this_mask - forced_masks[d]).coalesce())

    if not is_3d:
      masks_shape = [*true_mask.shape[0:3], 2]
    else:
      masks_shape = [*true_mask.shape[0:4], 2]

    true_values = true_mask.values()
    new_true_values = torch.cat([torch.zeros(size=true_values.shape, dtype=true_values.dtype), true_values], dim=-1)
    new_true_mask = torch.sparse_coo_tensor(true_mask.indices(), new_true_values,
                                            size=masks_shape, check_invariants=True)
    false_values = false_mask.values()
    new_false_values = torch.cat([false_values, torch.zeros(size=false_values.shape, dtype=false_values.dtype)], dim=-1)
    new_false_mask = torch.sparse_coo_tensor(false_mask.indices(), new_false_values,
                                            size=masks_shape, check_invariants=True)
    logit = (new_true_mask + new_false_mask).coalesce()
    logits.append(logit)

    this_mask = forced_masks[d]

    mask = mask - this_mask
    if uninteresting_masks is not None:
      mask = mask - uninteresting_masks[d]
    mask = eliminate_zero_or_less(mask)
  return logits
