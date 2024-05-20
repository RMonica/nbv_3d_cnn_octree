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

from octree_sparse_pytorch_utils import *

def load_octree_from_file(ifile):
  imgs = []
  masks = []
  uninteresting_masks = []
  image_pyramid = []
  sq_image_pyramid = []

  weighted_pyramid = None

  FIELD_OCTREE_LEVELS = 0
  FIELD_FOCUS_MASKS   = 1
  FIELD_IMAGE_PYRAMID = 2
  FIELD_WEIGHTED_IMAGE_PYRAMID = 3

  MAGIC_STRING_2D = "OCTREE2D"
  MAGIC_STRING_3D = "OCTREE3D"
  VERSION = 1

  magic_string = ifile.read(len(MAGIC_STRING_2D)).decode('ascii')
  if magic_string != MAGIC_STRING_2D and magic_string != MAGIC_STRING_3D:
    raise RuntimeError("load_octree_from_file: Magic string %s or %s expected, found %s" %
                       (MAGIC_STRING_2D, MAGIC_STRING_3D, magic_string))

  version = struct.unpack("Q", ifile.read(8))[0]
  if version != VERSION:
    raise RuntimeError("load_octree3d_from_file: expected version %d, found %d instead" % (VERSION, version))

  num_levels = struct.unpack("Q", ifile.read(8))[0]
  #print("num levels is %d" % num_levels)

  num_channels = struct.unpack("Q", ifile.read(8))[0]
  #print("num channels is %d" % num_channels)

  num_fields = struct.unpack("Q", ifile.read(8))[0]
  #print("num fields is %d" % num_fields)

  if magic_string == MAGIC_STRING_2D:
    ndims = 2
  else:
    ndims = 3

  for f in range(0, num_fields):
    field = struct.unpack("Q", ifile.read(8))[0]
    #print("FIELD: " + str(field))
    if field == FIELD_OCTREE_LEVELS:
      #print("parsing FIELD_OCTREE_LEVELS")
      for l in range(0, num_levels):
        #print("parsing level %d" % l)
        if ndims >= 3:
          depth = struct.unpack("Q", ifile.read(8))[0]
        height = struct.unpack("Q", ifile.read(8))[0]
        width = struct.unpack("Q", ifile.read(8))[0]

        size = struct.unpack("Q", ifile.read(8))[0]

        indices = np.fromfile(ifile, dtype=np.int64, count=size * ndims)
        indices = indices.reshape([ndims, size])
        indices = np.concatenate([np.zeros([1, size], dtype=np.int64), indices], axis=0)
        indices = torch.tensor(indices, dtype=torch.int64)

        values = np.fromfile(ifile, dtype=np.float32, count=size * num_channels)
        values = values.reshape([size, num_channels])
        values = torch.tensor(values, dtype=torch.float32)

        ones = np.ones(shape=[size, 1], dtype=np.float32)
        ones = torch.tensor(ones, dtype=torch.float32)

        img_size = [1, height, width]
        if ndims >= 3:
          img_size.append(depth)
        img_size.append(num_channels)

        #print("image size %s, sparse size is %d" % (str(img_size), size))

        mask_size = [1, height, width]
        if ndims >= 3:
          mask_size.append(depth)
        mask_size.append(1)

        #print("mask size %s, sparse size is %d" % (str(mask_size), size))

        img = torch.sparse_coo_tensor(indices, values, size=img_size, dtype=torch.float32, check_invariants=True).coalesce()
        mask = torch.sparse_coo_tensor(indices, ones, size=mask_size, dtype=torch.float32, check_invariants=True).coalesce()
        imgs.append(img)
        masks.append(mask)
        pass
      pass
    elif field == FIELD_FOCUS_MASKS:
      #print("parsing FIELD_FOCUS_MASKS")
      for l in range(0, num_levels):
        #print("parsing level %d" % l)
        if ndims >= 3:
          depth = struct.unpack("Q", ifile.read(8))[0]
        height = struct.unpack("Q", ifile.read(8))[0]
        width = struct.unpack("Q", ifile.read(8))[0]

        size = struct.unpack("Q", ifile.read(8))[0]

        indices = np.fromfile(ifile, dtype=np.int64, count=size * ndims)
        indices = indices.reshape([ndims, size])
        indices = np.concatenate([np.zeros([1, size], dtype=np.int64), indices], axis=0)
        indices = torch.tensor(indices, dtype=torch.int64)

        ones = np.ones(shape=[size, 1], dtype=np.float32)
        ones = torch.tensor(ones, dtype=torch.float32)

        mask_size = [1, height, width]
        if ndims >= 3:
          mask_size.append(depth)
        mask_size.append(1)

        #print("mask size %s, sparse size is %d" % (str(mask_size), size))

        mask = torch.sparse_coo_tensor(indices, ones, size=mask_size, dtype=torch.float32, check_invariants=True).coalesce()
        uninteresting_masks.append(mask)
        pass
      pass
    elif field == FIELD_IMAGE_PYRAMID:
      #print("parsing FIELD_IMAGE_PYRAMID")
      for l in range(0, num_levels):
        #print("parsing level %d" % l)
        if ndims >= 3:
          depth = struct.unpack("Q", ifile.read(8))[0]
        height = struct.unpack("Q", ifile.read(8))[0]
        width = struct.unpack("Q", ifile.read(8))[0]

        size = struct.unpack("Q", ifile.read(8))[0]

        indices = np.fromfile(ifile, dtype=np.int64, count=size * ndims)
        indices = indices.reshape([ndims, size])
        indices = np.concatenate([np.zeros([1, size], dtype=np.int64), indices], axis=0)
        indices = torch.tensor(indices, dtype=torch.int64)

        values = np.fromfile(ifile, dtype=np.float32, count=size * 2)
        values = values.reshape([size, 2])
        img_values = np.expand_dims(values[:, 0], -1)
        sq_img_values = np.expand_dims(values[:, 1], -1)
        img_values = torch.tensor(img_values, dtype=torch.float32)
        sq_img_values = torch.tensor(sq_img_values, dtype=torch.float32)

        img_size = [1, height, width]
        if ndims >= 3:
          img_size.append(depth)
        img_size.append(1)

        #print("mask size %s, sparse size is %dx2" % (str(img_size), size))

        image = torch.sparse_coo_tensor(indices, img_values, size=img_size, dtype=torch.float32,
                                        check_invariants=True).coalesce().detach()
        sq_image = torch.sparse_coo_tensor(indices, sq_img_values, size=img_size, dtype=torch.float32,
                                           check_invariants=True).coalesce().detach()

        image_pyramid.append(image)
        sq_image_pyramid.append(sq_image)
        pass
    elif field == FIELD_WEIGHTED_IMAGE_PYRAMID:
      #print("parsing FIELD_WEIGHTED_IMAGE_PYRAMID")
      images = []
      sq_images = []
      weightss = []
      for l in range(0, num_levels):
        #print("parsing level %d" % l)
        if ndims >= 3:
          depth = struct.unpack("Q", ifile.read(8))[0]
        height = struct.unpack("Q", ifile.read(8))[0]
        width = struct.unpack("Q", ifile.read(8))[0]

        size = struct.unpack("Q", ifile.read(8))[0]

        indices = np.fromfile(ifile, dtype=np.int64, count=size * ndims)
        indices = indices.reshape([ndims, size])
        indices = np.concatenate([np.zeros([1, size], dtype=np.int64), indices], axis=0)
        indices = torch.tensor(indices, dtype=torch.int64)

        values = np.fromfile(ifile, dtype=np.float32, count=size * 3)
        values = values.reshape([size, 3])
        img_values = np.expand_dims(values[:, 0], -1)
        sq_img_values = np.expand_dims(values[:, 1], -1)
        weight_values = np.expand_dims(values[:, 2], -1)
        img_values = torch.tensor(img_values, dtype=torch.float32)
        sq_img_values = torch.tensor(sq_img_values, dtype=torch.float32)
        weight_values = torch.tensor(weight_values, dtype=torch.float32)

        img_size = [1, height, width]
        if ndims >= 3:
          img_size.append(depth)
        img_size.append(1)

        #print("mask size %s, sparse size is %dx2" % (str(img_size), size))

        image = torch.sparse_coo_tensor(indices, img_values, size=img_size, dtype=torch.float32,
                                        check_invariants=True).coalesce().detach()
        sq_image = torch.sparse_coo_tensor(indices, sq_img_values, size=img_size, dtype=torch.float32,
                                           check_invariants=True).coalesce().detach()
        weights = torch.sparse_coo_tensor(indices, weight_values, size=img_size, dtype=torch.float32,
                                           check_invariants=True).coalesce().detach()

        images.append(image)
        sq_images.append(sq_image)
        weightss.append(weights)
        pass
      weighted_pyramid = (images, sq_images, weightss)
      pass
    else:
      rospy.logfatal("load_octree_from_file: unknown field %d" % field)
      exit(1)

  return imgs, masks, uninteresting_masks, image_pyramid, sq_image_pyramid, weighted_pyramid

def save_octree_to_file(ofile, imgs, masks, uninteresting_masks=None):

  FIELD_OCTREE_LEVELS = 0
  FIELD_FOCUS_MASKS   = 1

  if len(imgs[0].shape) != 4 and len(imgs[0].shape) != 5:
    print("save_octree_to_file: error: image shape is %s, expected 4 elements (for 2D) or 5 (for 3D)." % str(list(imgs[0].shape)))
    return False
  is_3d = (len(imgs[0].shape) == 5)

  MAGIC_STRING_2D = "OCTREE2D"
  MAGIC_STRING_3D = "OCTREE3D"
  MAGIC_STRING = MAGIC_STRING_2D if not is_3d else MAGIC_STRING_3D
  VERSION = 1

  ofile.write(MAGIC_STRING.encode('ascii'))

  ofile.write(VERSION.to_bytes(8, byteorder='little', signed=False))

  num_levels = len(imgs)
  num_channels = imgs[0].shape[-1]

  if len(masks) != num_levels:
    print("save_octree_to_file: error: number of masks does not match images.")
    return False

  if (uninteresting_masks is not None) and (len(uninteresting_masks) != num_levels):
    print("save_octree_to_file: error: number of uninteresting_masks does not match images.")
    return False

  ofile.write(num_levels.to_bytes(8, byteorder='little', signed=False))
  ofile.write(num_channels.to_bytes(8, byteorder='little', signed=False))

  num_fields = 1
  if uninteresting_masks is not None:
    num_fields += 1

  ofile.write(num_fields.to_bytes(8, byteorder='little', signed=False))

  if not is_3d:
    ndims = 2
  else:
    ndims = 3

  # FIELD_OCTREE_LEVELS
  ofile.write(FIELD_OCTREE_LEVELS.to_bytes(8, byteorder='little', signed=False))

  for l in range(0, num_levels):
    img = imgs[l] * masks[l]

    width = img.shape[2]
    height = img.shape[1]
    if is_3d:
      width = img.shape[3]
      height = img.shape[2]
      depth = img.shape[1]
      ofile.write(depth.to_bytes(8, byteorder='little', signed=False))
    ofile.write(height.to_bytes(8, byteorder='little', signed=False))
    ofile.write(width.to_bytes(8, byteorder='little', signed=False))

    indices = img.indices().detach().cpu().numpy().astype(np.uint64)
    values = img.values().detach().cpu().numpy().astype(np.float32)

    size = len(indices[0])
    ofile.write(size.to_bytes(8, byteorder='little', signed=False))

    indices = indices[1:] # remove first dimension (batch)
    indices.tofile(ofile)
    values.tofile(ofile)
    pass # FIELD_OCTREE_LEVELS

  # FIELD_FOCUS_MASKS
  if uninteresting_masks is not None:
    ofile.write(FIELD_FOCUS_MASKS.to_bytes(8, byteorder='little', signed=False))

    for l in range(0, num_levels):
      img = uninteresting_masks[l]

      width = img.shape[2]
      height = img.shape[1]
      if is_3d:
        width = img.shape[3]
        height = img.shape[2]
        depth = img.shape[1]
        ofile.write(depth.to_bytes(8, byteorder='little', signed=False))
      ofile.write(height.to_bytes(8, byteorder='little', signed=False))
      ofile.write(width.to_bytes(8, byteorder='little', signed=False))

      indices = img.indices().detach().cpu().numpy().astype(np.uint64)

      size = len(indices[0])
      ofile.write(size.to_bytes(8, byteorder='little', signed=False))

      indices = indices[1:]
      indices.tofile(ofile)
      pass
    pass # FIELD_FOCUS_MASK
  return True

def select_nth_channel(img, channel):
  return image_channels_first(img)[channel]

def octree_to_pytorch_image(outputs, output_masks, crop):
  img = octree_to_image(outputs, output_masks)
  if not (crop is None):
    img = image_crop(img, crop)
  img = image_channels_first(img)
  return torch.tensor(img, dtype=torch.float32)

def octree_to_image(outputs, output_masks):
  is_3d = (len(outputs[0].shape) == 5);

  result = outputs[0].to_dense().numpy(force=True)
  for i in range(1, len(outputs)):
    result = result.repeat(2, axis=1).repeat(2, axis=2)
    if is_3d:
      result = result.repeat(2, axis=3)
    result = result + outputs[i].to_dense().numpy(force=True)
  return result[0]

def image_crop(img, crop):
  if len(crop) == 2:
    return img[:crop[1], :crop[0]]
  elif len(crop) == 3:
    return img[:crop[2], :crop[1], :crop[0]]
  else:
    rospy.logfatal("octree_save_load: image_crop: unsupported number of dimensions: %s" % (str(crop)))

# transpose image so that the channels come first
def image_channels_first(img):
  num_channels = len(img.shape)
  tr_list = list(range(0, num_channels - 1))
  tr_list = [num_channels - 1, ] + tr_list
  return img.transpose(tr_list)

# transpose image so that the channels come last
def image_channels_last(img):
  num_channels = len(img.shape)
  tr_list = list(range(1, num_channels))
  tr_list = tr_list + [0, ]
  return img.transpose(tr_list)

def save_voxelgrid(outfilename, npmatrix):
  ofile = open(outfilename + ".binvoxelgrid", "wb")
  ofile.write(bytes("VXGR", 'utf-8'))

  version = 1
  width = npmatrix.shape[2]
  height = npmatrix.shape[1]
  depth = npmatrix.shape[0]
  metadata = np.asarray([version, width, height, depth], dtype=np.uint32)
  ofile.write(metadata.tobytes())

  npmatrix = npmatrix.astype('float32')
  ofile.write(npmatrix.tobytes())

  ofile.close()
  pass

