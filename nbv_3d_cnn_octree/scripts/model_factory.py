#!/usr/bin/python3

import enum_config

import octree_sparse_pytorch_enc_dec as sparse_enc_dec
import octree_sparse_minkowski_enc_dec as minkowski_enc_dec
import octree_pytorch_enc_dec as enc_dec

def get_model(model_type, engine, nin, nout, octree_depth, resblock_num, base_channels,
              label_hidden_channels, max_channels, is_3d):
  model = None

  engine_is_torchsparse = (engine == enum_config.EngineType.TORCHSPARSE)
  engine_is_pytorch = (engine == enum_config.EngineType.PYTORCH)
  engine_is_minkowski = (engine == enum_config.EngineType.MINKOWSKI)
  if model_type == enum_config.ModelType.ENC_DEC:
    model = enc_dec.EncoderDecoder(nin=2, nout=1, depth=octree_depth, base_channels=base_channels, max_channels=max_channels, is_3d=is_3d)
  if (model_type == enum_config.ModelType.SPARSE_ENC_DEC) and (engine_is_torchsparse or engine_is_pytorch):
    model = sparse_enc_dec.OctreeEncoderDecoder(nin=2, nout=1, depth=octree_depth, base_channels=base_channels, max_channels=max_channels,
                                                label_hidden_channels=label_hidden_channels, is_3d=is_3d,
                                                use_torchsparse=engine_is_torchsparse)
  if (model_type == enum_config.ModelType.SPARSE_ENC_DEC) and (engine_is_minkowski):
    model = minkowski_enc_dec.OctreeEncoderDecoder(nin=2, nout=1, depth=octree_depth, base_channels=base_channels,
                                                   max_channels=max_channels,
                                                   label_hidden_channels=label_hidden_channels, is_3d=is_3d)

  if model_type == enum_config.ModelType.RESNET:
    model = enc_dec.EncoderDecoder(nin=2, nout=1, depth=octree_depth, resblock_num=resblock_num,
                                   base_channels=base_channels, max_channels=max_channels,
                                   is_3d=is_3d,
                                   arch_type=enum_config.ArchType.RESBLOCK)
  if (model_type == enum_config.ModelType.SPARSE_RESNET) and (engine_is_torchsparse or engine_is_pytorch):
    model = sparse_enc_dec.OctreeEncoderDecoder(nin=2, nout=1, depth=octree_depth, resblock_num = resblock_num,
                                                base_channels=base_channels, max_channels=max_channels,
                                                label_hidden_channels=label_hidden_channels,
                                                is_3d=is_3d, use_torchsparse=engine_is_torchsparse,
                                                arch_type=enum_config.ArchType.RESBLOCK)
  if (model_type == enum_config.ModelType.SPARSE_RESNET) and (engine_is_minkowski):
    model = minkowski_enc_dec.OctreeEncoderDecoder(nin=2, nout=1, depth=octree_depth, base_channels=base_channels,
                                                   max_channels=max_channels,
                                                   label_hidden_channels=label_hidden_channels, is_3d=is_3d,
                                                   arch_type=enum_config.ArchType.RESBLOCK)

  return model
