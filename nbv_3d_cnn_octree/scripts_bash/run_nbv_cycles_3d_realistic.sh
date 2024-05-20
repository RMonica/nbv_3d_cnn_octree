#!/bin/bash

for env_i in {120..160}
  do
  roslaunch nbv_3d_cnn_octree simulate_nbv_cycle_3d_realistic.launch image_index:="$env_i" output_prefix:="${env_i}_" save_images:="false"
  done
