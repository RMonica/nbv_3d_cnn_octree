#!/usr/bin/python3

from enum import Enum

class ModelType(Enum):
  SPARSE_RESNET  = "sparse_resnet"
  SPARSE_ENC_DEC = "sparse_enc_dec"
  RESNET         = "resnet"
  ENC_DEC        = "enc_dec"
  pass

class InputType(Enum):
  OCTREE = "octree"
  IMAGE  = "image"
  pass

class EngineType(Enum):
  PYTORCH     = "pytorch"
  TORCHSPARSE = "torchsparse"
  MINKOWSKI   = "minkowski"
  pass

class ArchType(Enum):
  BASE        = "base"
  RESBLOCK    = "resblock"
  pass
