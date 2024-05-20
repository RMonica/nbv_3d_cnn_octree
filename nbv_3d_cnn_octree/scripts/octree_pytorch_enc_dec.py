#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
import pickle

import torch

import enum_config

from octree_sparse_pytorch_common import Resblock
import model_factory

class Encoder(torch.nn.Module):
  def __init__(self, nin, channels_list, depth=6, resblock_num=3, is_3d=False, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin

    self.is_3d = is_3d

    self.resblock_num = resblock_num

    self.arch_type = arch_type

    if self.is_3d:
      self.convtype = torch.nn.Conv3d
      self.transposeconvtype = torch.nn.ConvTranspose3d
    else:
      self.convtype = torch.nn.Conv2d
      self.transposeconvtype = torch.nn.ConvTranspose2d

    self.resblock_num = resblock_num
    self.depth = depth
    channels = channels_list

    use_batch_norm = False

    registered_blocks = []

    input_conv = self.convtype(self.nin, channels[self.depth - 1], 3, padding='same', dtype=torch.float32)
    prev_channels = channels[self.depth - 1]
    self.input_conv = input_conv

    self.convs = torch.nn.ModuleList([None] * (self.depth))
    self.convs2 = torch.nn.ModuleList([None] * (self.depth))

    if self.arch_type == enum_config.ArchType.BASE:
      for d in range(self.depth - 1, 0, -1):
        conv = self.convtype(prev_channels, prev_channels, 3, stride=1, padding='same', dtype=torch.float32)
        self.convs[d] = conv

        conv = self.convtype(prev_channels, channels[d - 1], 3, stride=2, dtype=torch.float32)
        prev_channels = channels[d - 1]
        self.convs2[d] = conv

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, self.depth)])
      for d in range(self.depth - 1, 0, -1):
        for i in range(0, self.resblock_num):
          resblock = Resblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d)
          self.resblocks[d].append(resblock)

        conv = self.convtype(prev_channels, channels[d - 1], 3, stride=2, dtype=torch.float32)
        prev_channels = channels[d - 1]
        self.convs[d] = conv

    self.nout = prev_channels
    pass

  def get_nout(self):
    return self.nout

  def forward_block(self, d, x):
    if self.arch_type == enum_config.ArchType.BASE:
      x = self.convs[d](x)
      x = torch.nn.functional.leaky_relu(x)

      x = self.convs2[d](x)
      x = torch.nn.functional.leaky_relu(x)
      pass

    if self.arch_type == enum_config.ArchType.RESBLOCK:
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x)

      x = self.convs[d](x)
      x = torch.nn.functional.leaky_relu(x)
    return x

  def forward(self, input):
    x = input
    skips = [None] * (self.depth)

    x = self.input_conv(x)
    x = torch.nn.functional.leaky_relu(x)

    for d in range(self.depth - 1, 0, -1):
      skips[d] = x
      x = self.forward_block(d, x)

    return x, skips
  pass

class Decoder(torch.nn.Module):
  def __init__(self, nin, nout, channels_list, depth=6, output_activation=torch.nn.functional.sigmoid, resblock_num=3,
               is_3d=False, arch_type=enum_config.ArchType.BASE):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.resblock_num = resblock_num

    self.is_3d = is_3d

    self.arch_type = arch_type

    if self.is_3d:
      self.convtype = torch.nn.Conv3d
      self.transposeconvtype = torch.nn.ConvTranspose3d
    else:
      self.convtype = torch.nn.Conv2d
      self.transposeconvtype = torch.nn.ConvTranspose2d

    self.resblock_num = resblock_num
    self.depth = depth
    channels = channels_list

    self.output_activation = output_activation

    use_batch_norm = False

    self.deconvs = torch.nn.ModuleList([None] * (self.depth))
    if self.arch_type == enum_config.ArchType.RESBLOCK:
      self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, self.depth)])

    prev_channels = self.nin
    for d in range(0, self.depth):
      if d > 0:
        deconv = self.transposeconvtype(prev_channels, channels[d], 3, stride=2, dtype=torch.float32)
        prev_channels = channels[d]
        self.deconvs[d] = deconv

      if self.arch_type == enum_config.ArchType.RESBLOCK:
        for i in range(0, self.resblock_num):
          resblock = Resblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d)
          self.resblocks[d].append(resblock)

    self.output_conv = self.convtype(prev_channels, self.nout, 1, padding='same', dtype=torch.float32)
    pass

  def forward_block(self, d, x):
    if self.arch_type == enum_config.ArchType.RESBLOCK:
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x)
    return x

  def forward(self, input, skips):
    x = input

    for d in range(0, self.depth):
      if d > 0:
        x = self.deconvs[d](x)
        x = torch.nn.functional.leaky_relu(x)
        skip = skips[d]
        padding = [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]]
        if (self.is_3d):
          padding = padding + [0, skip.shape[-3] - x.shape[-3]]
        if padding != [0, 0, 0, 0] and padding != [0, 0, 0, 0, 0, 0]:
          x = torch.nn.functional.pad(x, padding, mode='replicate')
        x = (x + skip) / 2.0

      x = self.forward_block(d, x)

    x = self.output_conv(x)
    x = self.output_activation(x)
    return x
  pass

class EncoderDecoder(torch.nn.Module):
  def __init__(self, nin, nout=1, depth=6, base_channels=16, max_channels=1000000000, is_3d=False, arch_type=enum_config.ArchType.BASE,
               resblock_num=3):
    super().__init__()

    self.nin = nin
    self.nout = nout

    channels = [min(base_channels*(2**(i-1)), max_channels) for i in range(depth, 0, -1)]

    self.encoder_net = Encoder(nin=self.nin, depth=depth, is_3d=is_3d, channels_list=channels, arch_type=arch_type,
                               resblock_num=resblock_num)
    self.decoder_net = Decoder(nin=self.encoder_net.get_nout(), nout=self.nout, depth=depth, is_3d=is_3d, channels_list=channels,
                               arch_type=arch_type, resblock_num=resblock_num)

  def forward(self, input):
    x, skips = self.encoder_net(input)
    output = self.decoder_net(x, skips)
    return output
