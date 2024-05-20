#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
import pickle

import torch

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

    channelb = int(self.nout // bottleneck)

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

class Encoder(torch.nn.Module):
  def __init__(self, nin, depth=6, base_channels=16, resblock_num=3, is_3d=False):
    super().__init__()

    self.nin = nin

    self.is_3d = is_3d

    if self.is_3d:
      self.convtype = torch.nn.Conv3d
    else:
      self.convtype = torch.nn.Conv2d

    self.resblock_num = resblock_num
    self.depth = depth
    channels = [base_channels*(2**(i-1)) for i in range(depth, 0, -1)]

    use_batch_norm = False

    registered_blocks = []

    input_conv = self.convtype(self.nin, channels[self.depth - 1], 3, padding='same', dtype=torch.float32)
    prev_channels = channels[self.depth - 1]
    self.input_conv = input_conv

    self.convs = torch.nn.ModuleList([None] * (self.depth))

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

  def forward(self, input):
    x = input
    skips = [None] * (self.depth)

    x = self.input_conv(x)
    x = torch.nn.functional.leaky_relu(x)

    for d in range(self.depth - 1, 0, -1):
      skips[d] = x
      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x)

      x = self.convs[d](x)
      x = torch.nn.functional.leaky_relu(x)
    return x, skips
  pass

class Decoder(torch.nn.Module):
  def __init__(self, nin, nout, depth=6, base_channels=16, output_activation=torch.nn.functional.sigmoid, resblock_num=3,
               is_3d=False):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.is_3d = is_3d

    if self.is_3d:
      self.convtype = torch.nn.Conv3d
      self.transposeconvtype = torch.nn.ConvTranspose3d
    else:
      self.convtype = torch.nn.Conv2d
      self.transposeconvtype = torch.nn.ConvTranspose2d

    self.resblock_num = resblock_num
    self.depth = depth
    channels = [base_channels*(2**(i-1)) for i in range(depth, 0, -1)]

    self.output_activation = output_activation

    use_batch_norm = False

    self.deconvs = torch.nn.ModuleList([None] * (self.depth))
    self.resblocks = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(0, self.depth)])

    prev_channels = self.nin
    for d in range(0, self.depth):
      if d > 0:
        deconv = self.transposeconvtype(prev_channels, channels[d], 3, stride=2, dtype=torch.float32)
        prev_channels = channels[d]
        self.deconvs[d] = deconv

      for i in range(0, self.resblock_num):
        resblock = Resblock(nin=prev_channels, nout=prev_channels, use_batch_norm=use_batch_norm, is_3d=self.is_3d)
        self.resblocks[d].append(resblock)

    self.output_conv = self.convtype(prev_channels, self.nout, 1, padding='same', dtype=torch.float32)
    pass

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

      for i in range(0, self.resblock_num):
        x = self.resblocks[d][i](x)

    x = self.output_conv(x)
    x = self.output_activation(x)
    return x
  pass

class EncoderDecoder(torch.nn.Module):
  def __init__(self, nin, nout=1, depth=6, base_channels=16, resblock_num=3, is_3d=False):
    super().__init__()

    self.nin = nin
    self.nout = nout

    self.is_3d = is_3d

    self.encoder_net = Encoder(nin=self.nin, depth=depth, resblock_num=resblock_num, is_3d=self.is_3d, base_channels=base_channels)
    self.decoder_net = Decoder(nin=self.encoder_net.get_nout(), nout=self.nout, depth=depth, resblock_num=resblock_num,
                               is_3d=self.is_3d, base_channels=base_channels)

  def forward(self, input):
    x, skips = self.encoder_net(input)
    output = self.decoder_net(x, skips)
    return output
