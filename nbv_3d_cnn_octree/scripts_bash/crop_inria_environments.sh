#!/bin/bash -i

# random seed
RANDOM=11

height=1250
width=1250
crop_height=400
crop_width=400

roscd nbv_3d_cnn
if [ $? -ne 0 ]
  then
  exit 1
  fi
INDIR="`pwd`/data/inria_environments/"
roscd nbv_3d_cnn_octree
if [ $? -ne 0 ]
  then
  exit 1
  fi
OUTDIR="`pwd`/data/inria_environments_400/"

for i in {0..179}
  do
  random_x_crop=$((RANDOM % (height - crop_height)))
  random_y_crop=$((RANDOM % (width - crop_width)))
  echo "${INDIR}${i}_environment.png" -crop "${crop_width}x${crop_height}+${random_x_crop}+${random_y_crop}" "${OUTDIR}${i}_environment.png"
  convert "${INDIR}${i}_environment.png" -crop "${crop_width}x${crop_height}+${random_x_crop}+${random_y_crop}" "${OUTDIR}${i}_environment.png"
  done



